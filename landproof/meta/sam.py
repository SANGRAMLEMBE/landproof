"""SAM 3.1 evidence backends.

* MetaApiSam   - sam-3.1 on Meta Model API (POST /v1/responses), parsed with Meta's
                 official `meta-sam-parser`. This is the production backend.
* LocalSam     - SAM 3 / 3.1 open weights (facebook/sam3.1) on your own GPU or Modal.
* OracleSam    - ground-truth label maps, optionally degraded (missed small objects,
                 dropped instances). ONLY for offline tests; every Instance it returns is
                 tagged backend="oracle" and every certificate inherits that tag, so
                 oracle results can never be mistaken for SAM results.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ..evidence import Instance, frame_digest
from .client import METER, openai_client, to_data_url, with_retries


@dataclass
class Region:
    """Crop (x0, y0, x1, y1) of the parent frame, resized so its long side = out_px."""

    x0: int
    y0: int
    x1: int
    y1: int
    out_px: int | None = None

    def clip(self, w: int, h: int) -> "Region":
        return Region(max(0, self.x0), max(0, self.y0), min(w, self.x1), min(h, self.y1), self.out_px)

    @property
    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0

    def scale(self) -> float:
        if not self.out_px:
            return 1.0
        return self.out_px / max(self.size)


class SamBackend:
    name = "base"

    def segment(self, image: np.ndarray, phrase: str, region: Region | None = None) -> list[Instance]:
        h, w = image.shape[:2]
        if region is None:
            return self._segment_array(image, phrase, None)
        r = region.clip(w, h)
        crop = image[r.y0 : r.y1, r.x0 : r.x1]
        s = r.scale()
        if s != 1.0:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        insts = self._segment_array(crop, phrase, (image, r, s))
        return [i.transformed((r.x0, r.y0), s) for i in insts]

    def _segment_array(self, image: np.ndarray, phrase: str, parent) -> list[Instance]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------- parsing
def parse_sam_output(text: str, video: bool = False, phrase: str = "", backend: str = "sam-3.1") -> list[Instance]:
    """Parse SAM 3.1 special-token output_text into Instances (box-local rasters)."""
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format

    text = (text or "").strip()
    if not text:
        return []  # "no match" is a normal, valid result
    fmt = video_segmentation_format() if video else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text)
    fin = parser.finish(CompletedOutcome())
    result = fin["result"] if isinstance(fin, dict) else fin.result
    box_conf = {}
    for rec in result.records:
        if rec.kind == "box":
            key = (rec.frame.frame_index if rec.frame else None, rec.object_id)
            box_conf[key] = rec.confidence
    out = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        fidx = rec.frame.frame_index if rec.frame else None
        raw = decode_mask_to_raster(rec.mask)
        raster = np.frombuffer(raw, dtype=np.uint8).reshape(rec.mask.height, rec.mask.width).astype(bool)
        b = rec.bounds
        conf = rec.confidence if rec.confidence is not None else box_conf.get((fidx, rec.object_id))
        out.append(
            Instance(
                object_id=f"{phrase}#{rec.object_id}",
                phrase=phrase,
                bounds=(float(b.left), float(b.top), float(b.right), float(b.bottom)),
                raster=raster,
                confidence=conf,
                frame_index=fidx,
                backend=backend,
            )
        )
    return out


# --------------------------------------------------------------------- Meta API
class MetaApiSam(SamBackend):
    name = "sam-3.1@meta-api"

    def __init__(self, model: str = "sam-3.1", cache_dir: str | Path | None = "cache/sam", max_side: int = 2048, mask_encoding: str = "lossless"):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_side = max_side
        self.mask_encoding = mask_encoding
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = openai_client()
        return self._client

    def _cache_path(self, image: np.ndarray, phrase: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256((frame_digest(image) + "|" + phrase + "|" + self.model + f"|{self.max_side}").encode()).hexdigest()
        return self.cache_dir / f"{key[:2]}/{key}.pkl"

    def _segment_array(self, image: np.ndarray, phrase: str, parent) -> list[Instance]:
        cp = self._cache_path(image, phrase)
        if cp and cp.exists():
            return pickle.loads(cp.read_bytes())
        h, w = image.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        send = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": phrase},
                        {"type": "input_image", "image_url": to_data_url(send)},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true", "mask_encoding": self.mask_encoding},
        }
        resp = with_retries(lambda: self.client.responses.create(**payload))
        METER.add("sam-3.1/image")
        text = _output_text(resp)
        insts = parse_sam_output(text, video=False, phrase=phrase, backend=self.name)
        if s < 1:
            insts = [i.transformed((0.0, 0.0), s) for i in insts]
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(pickle.dumps(insts))
        return insts

    def segment_video(self, video_path: str | Path, phrase: str) -> dict[int, list[Instance]]:
        """Track a concept through an MP4/MOV clip (<= 15k frames, <= 16 objects/frame)."""
        with open(video_path, "rb") as f:
            up = with_retries(lambda: self.client.files.create(file=f, purpose="user_data"))
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": phrase}, {"type": "input_video", "file_id": up.id}],
                }
            ],
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "stream": True,
        }

        def run():
            chunks = []
            for ev in self.client.responses.create(**payload):
                if getattr(ev, "type", "") == "response.output_text.delta":
                    chunks.append(getattr(ev, "delta", "") or "")
            return "".join(chunks)

        text = with_retries(run)
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        METER.add("sam-3.1/video_frame", max(50, n))
        by_frame: dict[int, list[Instance]] = {}
        for inst in parse_sam_output(text, video=True, phrase=phrase, backend=self.name):
            by_frame.setdefault(inst.frame_index or 0, []).append(inst)
        return by_frame


def _output_text(resp) -> str:
    text = getattr(resp, "output_text", "") or ""
    if text:
        return text
    parts = []
    for item in getattr(resp, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", "") == "output_text":
                parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


# --------------------------------------------------------------------- open weights
class LocalSam(SamBackend):
    """SAM 3 open weights via the facebookresearch/sam3 package (needs gated HF access)."""

    name = "sam3@open-weights"

    def __init__(self, device: str | None = None, threshold: float = 0.4):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam3_image_model(device=self.device)
        self.processor = Sam3Processor(self.model)
        self.threshold = threshold

    def _segment_array(self, image: np.ndarray, phrase: str, parent) -> list[Instance]:
        from PIL import Image as PILImage

        state = self.processor.set_image(PILImage.fromarray(image))
        out = self.processor.set_text_prompt(state=state, prompt=phrase)
        masks = out["masks"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        insts = []
        for k in range(len(scores)):
            if scores[k] < self.threshold:
                continue
            m = masks[k].squeeze().astype(bool)
            inst = Instance.from_full_mask(m, f"{phrase}#{k}", phrase, self.name, float(scores[k]))
            if inst:
                insts.append(inst)
        return insts


# --------------------------------------------------------------------- oracle (tests only)
class OracleSam(SamBackend):
    """Ground-truth evidence for offline tests. Never present its output as SAM output."""

    name = "oracle"

    def __init__(self, vocab, min_area_px: float = 0.0, miss_prob: float = 0.0, seed: int = 0):
        self.vocab = vocab
        self.labels: dict[str, np.ndarray] = {}
        self.min_area_px = min_area_px
        self.miss_prob = miss_prob
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def register(self, image: np.ndarray, label: np.ndarray) -> None:
        self.labels[frame_digest(image)] = label

    def _segment_array(self, image: np.ndarray, phrase: str, parent) -> list[Instance]:
        self.calls += 1
        if parent is None:
            label = self.labels.get(frame_digest(image))
        else:
            src, r, s = parent
            full = self.labels.get(frame_digest(src))
            label = None
            if full is not None:
                label = full[r.y0 : r.y1, r.x0 : r.x1]
                if s != 1.0:
                    label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if label is None:
            raise KeyError("OracleSam: frame not registered (did you call register(image, label)?)")
        cid = self.vocab.class_of(phrase).id
        comp, n = ndimage.label(label == cid)
        out = []
        for k in range(1, n + 1):
            m = comp == k
            if m.sum() < self.min_area_px or self.rng.random() < self.miss_prob:
                continue
            inst = Instance.from_full_mask(m, f"{phrase}#{k}", phrase, self.name, 1.0)
            if inst:
                out.append(inst)
        return out

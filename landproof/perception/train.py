"""Train the onboard student on (image, label) pairs -- SAM 3.1 pseudo-labels or ground truth.

Altitude augmentation: random crops at 0.5-1.6x scale emulate flying higher or lower,
and 90-degree rotations/flips make the model heading-invariant (nadir camera).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from ..masks import confusion_miou
from ..vocab import IGNORE
from .student import MEAN, STD, build_student


def _load_pair(img_path, lab_path):
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
    if lab.shape[:2] != rgb.shape[:2]:
        lab = cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb, lab


class SegDataset:
    def __init__(self, pairs, crop: int = 512, train: bool = True, cache: bool = True):
        self.pairs, self.crop, self.train = list(pairs), crop, train
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.pairs) * (4 if self.train else 1)

    def _get(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        item = self.pairs[i]
        rgb, lab = (item[0], item[1]) if isinstance(item[0], np.ndarray) else _load_pair(*item)
        if self._cache is not None:
            self._cache[i] = (rgb, lab)
        return rgb, lab

    def __getitem__(self, idx):
        import torch

        rgb, lab = self._get(idx % len(self.pairs))
        if self.train:
            s = random.uniform(0.5, 1.6) * self.crop / min(rgb.shape[:2]) * 1.5
            s = min(max(s, self.crop / min(rgb.shape[:2])), 4.0)
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            y = random.randint(0, rgb.shape[0] - self.crop)
            x = random.randint(0, rgb.shape[1] - self.crop)
            rgb, lab = rgb[y : y + self.crop, x : x + self.crop], lab[y : y + self.crop, x : x + self.crop]
            k = random.randint(0, 3)
            rgb, lab = np.rot90(rgb, k), np.rot90(lab, k)
            if random.random() < 0.5:
                rgb, lab = rgb[:, ::-1], lab[:, ::-1]
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15), 0, 255)
        else:
            s = 768 / max(rgb.shape[:2])
            if s < 1:
                rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                lab = cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            h, w = (rgb.shape[0] // 16) * 16, (rgb.shape[1] // 16) * 16
            rgb, lab = rgb[:h, :w], lab[:h, :w]
        x = ((np.ascontiguousarray(rgb).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64))


def class_weights(pairs, num_classes: int, max_samples: int = 200):
    counts = np.zeros(num_classes, np.float64)
    for item in pairs[:max_samples]:
        lab = item[1] if isinstance(item[1], np.ndarray) else cv2.imread(str(item[1]), cv2.IMREAD_UNCHANGED)
        v = lab[lab != IGNORE]
        counts += np.bincount(v.ravel(), minlength=num_classes)[:num_classes]
    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.sqrt(freq + 1e-4)
    return w / w.mean()


def evaluate(model, loader, num_classes, device) -> tuple[float, list]:
    import torch

    model.eval()
    preds, gts = [], []
    with torch.inference_mode():
        for x, y in loader:
            out = model(x.to(device))["out"].argmax(1).cpu().numpy()
            preds.append(out.ravel())
            gts.append(y.numpy().ravel())
    miou, per = confusion_miou(np.concatenate(preds), np.concatenate(gts), num_classes)
    return miou, [None if math.isnan(v) else round(float(v), 4) for v in per]


def train_student(train_pairs, val_pairs, num_classes: int, out_path: str | Path, epochs: int = 30, batch_size: int = 8, lr: float = 1e-3,
                  crop: int = 512, device: str | None = None, workers: int = 4, log_every: int = 20, extra_meta: dict | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student(num_classes).to(device)
    w = torch.tensor(class_weights(train_pairs, num_classes), dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tl = DataLoader(SegDataset(train_pairs, crop, True), batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    vl = DataLoader(SegDataset(val_pairs, crop, False), batch_size=1, shuffle=False, num_workers=min(2, workers))
    steps = epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, history, t0, step = -1.0, [], time.time(), 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = crit(model(x)["out"], y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        miou, per = evaluate(model, vl, num_classes, device) if val_pairs else (float("nan"), [])
        history.append({"epoch": ep, "val_miou": miou, "per_class": per})
        print(f"== epoch {ep} val mIoU {miou:.4f}", flush=True)
        if val_pairs == [] or miou > best:
            best = miou
            torch.save({"state_dict": model.state_dict(), "num_classes": num_classes, "val_miou": miou, "epoch": ep, "meta": extra_meta or {}}, out_path)
    report = {"best_val_miou": best, "history": history, "train_pairs": len(train_pairs), "val_pairs": len(val_pairs), "minutes": round((time.time() - t0) / 60, 1), "meta": extra_meta or {}}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return report


def measure_fps(checkpoint: str | Path, size=(720, 960), n: int = 50, device: str | None = None) -> dict:
    import torch

    from .student import TorchStudent

    st = TorchStudent(checkpoint, device=device)
    rgb = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
    for _ in range(5):
        st.predict(rgb)
    if st.device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        st.predict(rgb)
    if st.device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    return {"device": st.device, "ms_per_frame": round(dt * 1000, 2), "fps": round(1 / dt, 1), "input_px": st.input_px}

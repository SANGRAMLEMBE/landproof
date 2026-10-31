"""Train the student on a Modal GPU (A10G by default, ~$1.10/h).

  modal volume create landproof-data
  modal volume put landproof-data data/pseudo/floodnet_train /pseudo/floodnet_train
  modal volume put landproof-data cache/gt_floodnet_val /gt/floodnet_val
  modal run modal_apps/train_student.py --train /pseudo/floodnet_train --val /gt/floodnet_val --out /ckpt/student_sam.pt
  modal volume get landproof-data /ckpt/student_sam.pt checkpoints/
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch>=2.4", "torchvision>=0.19", "numpy", "scipy", "opencv-python-headless", "pillow", "pydantic>=2.6", "pyyaml", "tqdm",
                 "openai>=1.40", "meta-sam-parser>=0.1.0", "requests", "python-dotenv")
    .add_local_dir(ROOT / "configs", "/root/configs")
    .add_local_python_source("landproof")
)
data = modal.Volume.from_name("landproof-data", create_if_missing=True)
app = modal.App("landproof-train")


@app.function(image=image, gpu="A10G", timeout=4 * 60 * 60, volumes={"/data": data})
def train_remote(train_dir: str, val_dir: str, out: str, epochs: int = 40, batch_size: int = 16, extra_dirs: list[str] | None = None):
    from landproof.perception.train import measure_fps, train_student
    from landproof.vocab import Vocabulary

    def pairs(d):
        d = Path("/data" + d)
        return [(p, d / "labels" / (p.stem + ".png")) for p in sorted((d / "images").glob("*.jpg")) if (d / "labels" / (p.stem + ".png")).exists()]

    vocab = Vocabulary.load(Path("/root/configs/hazards.yaml"))
    tr = pairs(train_dir)
    for e in extra_dirs or []:
        tr += pairs(e)
    rep = train_student(tr, pairs(val_dir), vocab.num_classes, "/data" + out, epochs=epochs, batch_size=batch_size, workers=8,
                        extra_meta={"train_dir": train_dir, "extra": extra_dirs})
    rep["speed_a10g"] = measure_fps("/data" + out, device="cuda")
    data.commit()
    return {k: rep[k] for k in ("best_val_miou", "minutes", "speed_a10g")}


@app.local_entrypoint()
def main(train: str, val: str, out: str = "/ckpt/student.pt", epochs: int = 40, extra: str = ""):
    print(train_remote.remote(train, val, out, epochs, 16, [e for e in extra.split(",") if e]))

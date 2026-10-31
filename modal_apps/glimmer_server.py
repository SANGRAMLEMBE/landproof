"""Serve Muse Glimmer (open weights, Apache-2.0) as an OpenAI-compatible endpoint on Modal.

This is the "edge box" stand-in: in a real deployment Glimmer would run on a
Jetson AGX Orin 64GB-class companion computer or ground station, not a 250 g drone.

  pip install modal && modal setup
  modal secret create glimmer-api-key GLIMMER_API_KEY=$(openssl rand -hex 16)
  modal secret create huggingface HF_TOKEN=hf_...            # if the weights are gated for you
  modal deploy modal_apps/glimmer_server.py
  # .env:  GLIMMER_BASE_URL=https://<workspace>--landproof-glimmer-serve.modal.run/v1
  #        GLIMMER_API_KEY=<same key>

Defaults: FP8 block-scaled weights on one L40S (48 GB, ~$1.95/h). Set GLIMMER_GPU=H100
and GLIMMER_MODEL_REPO=meta-models/Muse-Glimmer-30B for BF16. The container scales to
zero after 10 idle minutes; the first request after that pays a cold start (minutes) -
warm it up before recording the demo.
"""

import os
import subprocess

import modal

MODEL_REPO = os.environ.get("GLIMMER_MODEL_REPO", "RedHatAI/Muse-Glimmer-30B-FP8-block")
GPU = os.environ.get("GLIMMER_GPU", "L40S")
SERVED_NAME = "muse-glimmer"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm>=0.28.0", "huggingface_hub[hf_transfer]>=0.30")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_DEEP_GEMM": "0", "VLLM_MOE_USE_DEEP_GEMM": "0"})
)
hf_cache = modal.Volume.from_name("landproof-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("landproof-vllm-cache", create_if_missing=True)

app = modal.App("landproof-glimmer")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("glimmer-api-key"), modal.Secret.from_name("huggingface", required_keys=[])],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_REPO,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["GLIMMER_API_KEY"],
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--generation-config", "auto",
        "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
        "--reasoning-parser", "muse_glimmer",
        "--limit-mm-per-prompt", '{"image": 2}',
    ]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def smoke_test():
    """modal run modal_apps/glimmer_server.py  -> one text + one image request."""
    import base64
    import io

    import numpy as np
    from openai import OpenAI
    from PIL import Image

    url = serve.get_web_url() + "/v1"
    client = OpenAI(base_url=url, api_key=os.environ.get("GLIMMER_API_KEY", "EMPTY"), timeout=1200)
    img = Image.fromarray((np.random.default_rng(0).random((256, 256, 3)) * 60 + 80).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.chat.completions.create(
        model=SERVED_NAME,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Describe this image in five words."},
                                               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]}],
        temperature=1.0, top_p=0.95, max_tokens=2048,
    )
    print(url, "->", r.choices[0].message.content)

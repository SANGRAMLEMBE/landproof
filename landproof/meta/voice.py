"""Human override by voice: Muse Voice Transcribe (realtime WebSocket) + multilingual command parser.

The operator's word outranks every model ("human > verifier > student > LLM"). Supported
override languages here: English, Tamil, Hindi (Muse Voice Transcribe also covers Telugu,
Kannada, Marathi, Bengali and more - extend COMMANDS).

Protocol (Meta Model API, realtime only): connect to wss://api.meta.ai/v1/asr/realtime,
send one `session.update` (PCM16 mono 16/24 kHz, optional language), stream
`input_audio_buffer.append` chunks (<= 4 s each, base64), commit, and read
`conversation.item.input_audio_transcription.completed` events.
EXPERIMENTAL: written from the public docs, not yet exercised against the live API -
if the handshake differs, `transcribe_wav` raises and you can pass --voice-text instead.
"""

from __future__ import annotations

import base64
import json
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

from ..budget import BUDGET
from .client import api_key, base_url

COMMANDS = {
    "abort": [
        # English
        r"\babort\b", r"\bstop\b", r"\bdon'?t land\b", r"\bdo not land\b", r"\bpull up\b", r"\bclimb\b", r"\bgo up\b", r"\bwait\b", r"\bhold\b",
        # Hindi (Devanagari + romanised)
        "रुको", "रोको", "मत उतारो", "ऊपर जाओ", r"\bruko\b", r"\broko\b", r"\bmat utaro\b",
        # Tamil (script + romanised)
        "நிறுத்து", "நில்", "இறங்காதே", "இறக்காதே", "மேல போ", "மேலே போ", r"\bniruthu\b", r"\bnil\b", r"\birangaathe\b", r"\bmela po\b",
    ],
}


def parse_command(text: str) -> str | None:
    """Map a transcript to an override command ('abort') or None."""
    t = unicodedata.normalize("NFC", (text or "").lower())
    for cmd, pats in COMMANDS.items():
        for p in pats:
            if re.search(p, t):
                return cmd
    return None


def _read_pcm16(path: str | Path, rate: int = 16000) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("WAV must be 16-bit PCM (ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 out.wav)")
    a = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != rate:  # simple linear resample
        x = np.arange(len(a)) / sr
        xi = np.arange(int(len(a) * rate / sr)) / rate
        a = np.interp(xi, x, a).astype(np.int16)
    return a.tobytes(), len(a) / rate


def transcribe_wav(path: str | Path, language: str | None = None, model: str = "muse-voice-transcribe-1.0", rate: int = 16000, timeout_s: float = 30.0) -> str:
    """Stream a short WAV to Muse Voice Transcribe and return the final transcript."""
    from websockets.sync.client import connect

    pcm, seconds = _read_pcm16(path, rate)
    BUDGET.reserve("muse-voice-transcribe-1.0/second", seconds)
    url = base_url().replace("https://", "wss://").replace("http://", "ws://") + f"/asr/realtime?model={model}"
    session = {"format": {"type": "audio/pcm", "rate": rate, "channels": 1}, "transcription": {"model": model}, "turn_detection": None}
    if language:
        session["transcription"]["language"] = language
    texts, deltas = [], []
    with connect(url, additional_headers={"Authorization": f"Bearer {api_key()}"}, open_timeout=15) as ws:
        ws.send(json.dumps({"type": "session.update", "session": session}))
        step = rate * 2 * 3  # 3-second chunks (limit is 4 s)
        for i in range(0, len(pcm), step):
            ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + step]).decode()}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                ev = json.loads(ws.recv(timeout=max(0.5, timeout_s - (time.time() - t0))))
            except TimeoutError:
                break
            typ = ev.get("type", "")
            if typ == "error":
                raise RuntimeError(f"Muse Voice Transcribe error: {ev}")
            if typ.endswith("input_audio_transcription.delta"):
                deltas.append(ev.get("delta", ""))
            if typ.endswith("input_audio_transcription.completed"):
                texts.append(ev.get("transcript", ""))
                break
    BUDGET.add("muse-voice-transcribe-1.0/second", seconds, note="voice override")
    return " ".join(texts).strip() or "".join(deltas).strip()

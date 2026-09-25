#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render the scripted user utterances used by ``headless.py`` with Piper TTS.

    pip install piper-tts
    python piper_utterances.py            # writes fixtures/*.wav (24 kHz mono PCM16)

The fixtures are committed; rerun only to change them.
"""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

import numpy as np

VOICE = "en_US-lessac-medium"
RATE = 24_000
HERE = Path(__file__).parent

UTTERANCES = {
    "hello": "Hello, can you hear me?",
    "capital": "What is the capital of France?",
    "population": "And what is its population?",
    "story": "Tell me a long story about a lighthouse keeper.",
    "interrupt": "Sorry, stop. What were you just saying?",
    "weather": "What is the weather like in Paris right now?",
    "mmhm": "Mm-hm.",
    "french": "Say something about the weather.",
}


def synthesize(voice, text: str) -> np.ndarray:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    if rate != RATE:
        n = int(len(pcm) * RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm)
    return pcm.astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice-dir", type=Path, default=HERE / "piper_voices")
    parser.add_argument("--out", type=Path, default=HERE / "fixtures")
    args = parser.parse_args()

    from piper.download_voices import download_voice
    from piper.voice import PiperVoice

    args.voice_dir.mkdir(parents=True, exist_ok=True)
    model = args.voice_dir / f"{VOICE}.onnx"
    if not model.exists():
        download_voice(VOICE, args.voice_dir)
    voice = PiperVoice.load(str(model))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in UTTERANCES.items():
        pcm = synthesize(voice, text)
        with wave.open(str(args.out / f"{name}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(pcm.tobytes())
        print(f"{name}.wav  {len(pcm) / RATE:.2f}s  {text}")


if __name__ == "__main__":
    main()

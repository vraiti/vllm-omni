#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deterministic, room-less LiveKit ``AgentSession`` against ``/v1/live/sessions``.

Plays a scripted list of user utterances in real time through ``GPTLiveModel``
and records everything:

* ``events.jsonl``: every WebSocket event in both directions (audio replaced by
  its size), captured by a local logging proxy between the plugin and vLLM-Omni;
* ``session.wav``: stereo, left = user audio as sent, right = agent audio at
  playout time;
* ``summary.json`` and a printed summary: per-utterance transcripts, first-audio
  latency, barge-in cut latency, tool calls.

    python headless.py --model <served name> --voice ethan --scenario conversation
    python headless.py --model <served name> --voice default --no-tools --scenario native

Scenarios are built in (``--list``) or read from a JSON file (``--script``): a
list of steps ``{"wav": name|path, "gap_s": s}``, ``{"silence_s": s}``, or
``{"wav": ..., "after_reply_s": s}`` (start ``s`` seconds after the agent's
reply to the previous utterance begins, to barge in).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from aiohttp import web
from livekit import rtc
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities
from livekit.plugins.openai.realtime import GPTLiveModel

HERE = Path(__file__).parent
RATE = 24_000
FRAME = RATE // 10  # 100 ms

SCENARIOS: dict[str, list[dict[str, Any]]] = {
    # 1: connect, one utterance, orderly close
    "connect": [{"silence_s": 2}, {"wav": "hello", "gap_s": 6}],
    # 5: single utterance
    "single": [{"silence_s": 1}, {"wav": "capital", "gap_s": 10}],
    # 6: three turns carrying context
    "conversation": [
        {"silence_s": 1},
        {"wav": "capital", "gap_s": 8},
        {"wav": "population", "gap_s": 8},
        {"wav": "hello", "gap_s": 8},
    ],
    # 7: barge-in 1 s into a long reply, then ask what was said
    "barge_in": [
        {"silence_s": 1},
        {"wav": "story", "gap_s": 0},
        {"wav": "interrupt", "after_reply_s": 1.0, "gap_s": 10},
    ],
    # 8: backchannel during a reply (needs --min-speech-duration-ms >= 400 on the VAD service)
    "backchannel": [
        {"silence_s": 1},
        {"wav": "story", "gap_s": 0},
        {"wav": "mmhm", "after_reply_s": 1.0, "gap_s": 15},
    ],
    # 11: tool call
    "tools": [{"silence_s": 1}, {"wav": "weather", "gap_s": 15}],
    # 9 (native models): continuous input with short and long overlaps
    "native": [
        {"silence_s": 2},
        {"wav": "story", "gap_s": 0},
        {"wav": "mmhm", "after_reply_s": 1.0, "gap_s": 0},
        {"wav": "interrupt", "after_reply_s": 4.0, "gap_s": 10},
    ],
    # 13: long conversation (run against an overlay with a small max_model_len)
    "long": [{"silence_s": 1}]
    + [{"wav": w, "gap_s": 7} for w in ["capital", "population", "hello", "weather", "story"] * 2],
    # 14: usage events (session.usage.updated every 60 s)
    "usage": [{"silence_s": 1}, {"wav": "hello", "gap_s": 70}],
    # 2: reconnect replay (serve with session_lifetime_s: 60)
    "reconnect": [{"silence_s": 1}, {"wav": "capital", "gap_s": 70}, {"wav": "population", "gap_s": 10}],
}


def load_wav(name: str) -> np.ndarray:
    path = Path(name)
    if not path.suffix:
        path = HERE / "fixtures" / f"{name}.wav"
    with wave.open(str(path)) as wf:
        rate, channels = wf.getframerate(), wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    if rate != RATE:
        n = int(len(pcm) * RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.int16)
    return pcm


class Clock:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.t0


# --------------------------------------------------------------------------- #
# Audio I/O                                                                   #
# --------------------------------------------------------------------------- #


class ScriptedAudioInput(AudioInput):
    """Real-time 100 ms frames: queued utterances, silence otherwise."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(label="scripted")
        self._clock = clock
        self._pending: list[np.ndarray] = []
        self.sent: list[tuple[float, np.ndarray]] = []
        self._next_at: float | None = None
        self.idle = asyncio.Event()
        self.idle.set()

    def play(self, pcm: np.ndarray) -> None:
        self._pending.append(pcm)
        self.idle.clear()

    async def __anext__(self) -> rtc.AudioFrame:
        now = self._clock.now()
        if self._next_at is None:
            self._next_at = now
        if self._next_at > now:
            await asyncio.sleep(self._next_at - now)
        self._next_at += 0.1
        if self._pending:
            head = self._pending[0]
            chunk, rest = head[:FRAME], head[FRAME:]
            if rest.size:
                self._pending[0] = rest
            else:
                self._pending.pop(0)
            if chunk.size < FRAME:
                chunk = np.pad(chunk, (0, FRAME - chunk.size))
            if not self._pending:
                self.idle.set()
        else:
            chunk = np.zeros(FRAME, dtype=np.int16)
        self.sent.append((self._clock.now(), chunk))
        return rtc.AudioFrame(chunk.tobytes(), RATE, 1, FRAME)


class CaptureAudioOutput(AudioOutput):
    """Records agent audio at its playout time; honours interruption."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(
            label="capture", next_in_chain=None, sample_rate=RATE, capabilities=AudioOutputCapabilities(pause=False)
        )
        self._clock = clock
        self.segments: list[tuple[float, np.ndarray]] = []
        self._play_end = 0.0
        self._pushed = 0.0
        self._started: float | None = None
        self._flush_handle: asyncio.TimerHandle | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        now = self._clock.now()
        start = max(now, self._play_end)
        pcm = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)[:, 0].copy()
        if frame.sample_rate != RATE:
            n = int(len(pcm) * RATE / frame.sample_rate)
            pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.int16)
        self.segments.append((start, pcm))
        self._play_end = start + len(pcm) / RATE
        self._pushed += frame.duration
        if self._started is None:
            self._started = time.time()
            self.on_playback_started(created_at=self._started)

    def flush(self) -> None:
        super().flush()
        if not self._pushed:
            return
        pushed = self._pushed
        delay = max(0.0, self._play_end - self._clock.now())

        def done() -> None:
            self._pushed, self._started = 0.0, None
            self.on_playback_finished(playback_position=pushed, interrupted=False, synchronized_transcript=None)

        self._flush_handle = asyncio.get_event_loop().call_later(delay, done)

    def clear_buffer(self) -> None:
        if not self._pushed:
            return
        now = self._clock.now()
        played = self._pushed - max(0.0, self._play_end - now)
        # Drop what had not played yet.
        kept = []
        for start, pcm in self.segments:
            if start >= now:
                continue
            end = start + len(pcm) / RATE
            kept.append((start, pcm if end <= now else pcm[: int((now - start) * RATE)]))
        self.segments = kept
        self._play_end = now
        if self._flush_handle:
            self._flush_handle.cancel()
        self._pushed, self._started = 0.0, None
        self.on_playback_finished(playback_position=max(0.0, played), interrupted=True, synchronized_transcript=None)


# --------------------------------------------------------------------------- #
# Logging proxy                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class EventLog:
    clock: Clock
    path: Path
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, direction: str, raw: str) -> dict[str, Any]:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            event = {"type": "<non-json>", "raw": raw[:200]}
        audio_key = {"session.input_audio.append": "audio", "session.output_audio.delta": "delta"}.get(
            event.get("type", "")
        )
        if audio_key and isinstance(event.get(audio_key), str):
            event[audio_key] = {"bytes": len(base64.b64decode(event[audio_key]))}
        record = {"t": round(self.clock.now(), 3), "dir": direction, **event}
        self.events.append(record)
        return record

    def write(self) -> None:
        with self.path.open("w") as f:
            for record in self.events:
                f.write(json.dumps(record) + "\n")


async def start_proxy(upstream: str, log: EventLog) -> tuple[web.AppRunner, int]:
    """``ws://127.0.0.1:<port>/v1/live/sessions`` -> ``upstream``, logging both directions."""

    async def handle(request: web.Request) -> web.WebSocketResponse:
        client = web.WebSocketResponse(max_msg_size=0)
        await client.prepare(request)
        headers = {k: v for k, v in request.headers.items() if k.lower() == "authorization"}
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(upstream, headers=headers, max_msg_size=0) as server,
        ):

            async def pump(src, dst, direction: str) -> None:
                async for msg in src:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        log.add(direction, msg.data)
                        await dst.send_str(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                log.events.append({"t": round(log.clock.now(), 3), "dir": direction, "type": "<closed>"})
                await dst.close()

            await asyncio.gather(pump(client, server, "client"), pump(server, client, "server"), return_exceptions=True)
        return client

    app = web.Application()
    app.router.add_get("/v1/live/sessions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, port


# --------------------------------------------------------------------------- #
# Session                                                                     #
# --------------------------------------------------------------------------- #


@function_tool
async def lookup_weather(context: RunContext, location: str) -> str:
    """Look up the current weather for a location.

    Args:
        location: The city or region to look up.
    """
    return f"The weather in {location} is 22 degrees Celsius and sunny."


def first_output_after(log: EventLog, t: float) -> float | None:
    for ev in log.events:
        if ev["t"] >= t and ev.get("type") == "session.output_audio.delta":
            return ev["t"]
    return None


def last_output_before(log: EventLog, t: float) -> float | None:
    times = [ev["t"] for ev in log.events if ev.get("type") == "session.output_audio.delta" and ev["t"] <= t]
    return times[-1] if times else None


async def run(args: argparse.Namespace) -> int:
    steps = json.loads(Path(args.script).read_text()) if args.script else SCENARIOS[args.scenario]
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    log = EventLog(clock, out / "events.jsonl")
    upstream = f"ws://{args.host}:{args.port}/v1/live/sessions"
    runner, proxy_port = await start_proxy(upstream, log)

    model = GPTLiveModel(
        base_url=f"http://127.0.0.1:{proxy_port}/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
        model=args.model,
        voice=args.voice,
        responses_options={"model": args.model},
    )
    session = AgentSession(llm=model)
    audio_in = ScriptedAudioInput(clock)
    audio_out = CaptureAudioOutput(clock)
    session.input.audio = audio_in
    session.output.audio = audio_out

    notes: list[dict[str, Any]] = []
    session.on(
        "user_input_transcribed",
        lambda ev: notes.append({"t": clock.now(), "user": ev.transcript, "final": ev.is_final}),
    )
    session.on(
        "conversation_item_added",
        lambda ev: notes.append(
            {"t": clock.now(), "item": getattr(ev.item, "role", None), "text": getattr(ev.item, "text_content", None)}
        ),
    )
    session.on(
        "function_tools_executed",
        lambda ev: notes.append({"t": clock.now(), "tools": [(c.name, c.arguments) for c in ev.function_calls]}),
    )
    session.on("error", lambda ev: notes.append({"t": clock.now(), "error": repr(ev.error)}))

    agent = Agent(
        instructions="You are a helpful voice assistant. Keep replies short and conversational.",
        tools=[] if args.no_tools else [lookup_weather],
    )
    await session.start(agent=agent)

    utterances: list[dict[str, Any]] = []
    for step in steps:
        if "silence_s" in step:
            await asyncio.sleep(step["silence_s"])
            continue
        if "after_reply_s" in step and utterances:
            prev_end = utterances[-1]["end"]
            deadline = clock.now() + 30
            while (reply := first_output_after(log, prev_end)) is None and clock.now() < deadline:
                await asyncio.sleep(0.05)
            if reply is not None:
                await asyncio.sleep(max(0.0, reply + step["after_reply_s"] - clock.now()))
        pcm = load_wav(step["wav"])
        start = clock.now()
        audio_in.play(pcm)
        await audio_in.idle.wait()
        utterances.append({"wav": step["wav"], "start": start, "end": clock.now(), "barge_in": "after_reply_s" in step})
        await asyncio.sleep(step.get("gap_s", 0))

    await session.aclose()
    await asyncio.sleep(0.5)
    await runner.cleanup()
    log.write()

    # Stereo WAV: left = user as sent, right = agent at playout time.
    total = max(
        [t + len(p) / RATE for t, p in audio_in.sent] + [t + len(p) / RATE for t, p in audio_out.segments] + [0.0]
    )
    stereo = np.zeros((int(total * RATE) + RATE, 2), dtype=np.int16)
    for channel, segments in ((0, audio_in.sent), (1, audio_out.segments)):
        for t, pcm in segments:
            i = int(t * RATE)
            stereo[i : i + len(pcm), channel] = pcm[: max(0, len(stereo) - i)]
    with wave.open(str(out / "session.wav"), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(stereo.tobytes())

    summary = {"scenario": args.script or args.scenario, "utterances": [], "notes": notes}
    for u in utterances:
        entry = {"wav": u["wav"], "start": round(u["start"], 2), "end": round(u["end"], 2)}
        first = first_output_after(log, u["end"])
        entry["first_audio_latency_s"] = None if first is None else round(first - u["end"], 2)
        if u["barge_in"]:
            # Agent audio stops arriving; judge the cut from the WAV's right channel too.
            last = last_output_before(log, u["start"] + 3.0)
            entry["last_audio_after_barge_in_s"] = None if last is None else round(last - u["start"], 2)
        entry["input_transcripts"] = [
            ev["delta"]
            for ev in log.events
            if ev.get("type") == "session.input_transcript.delta" and ev["t"] >= u["start"]
        ][:1]
        summary["utterances"].append(entry)
    summary["output_transcript"] = "".join(
        ev["delta"] for ev in log.events if ev.get("type") == "session.output_transcript.delta"
    )
    summary["errors"] = [ev for ev in log.events if ev.get("type") == "error"]
    summary["delegations"] = sum(1 for ev in log.events if ev.get("type") == "session.delegation.created")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "notes"}, indent=2))
    print(f"\nartifacts: {out}")
    return 1 if summary["errors"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("VLLM_OMNI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_OMNI_PORT", "8000")))
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--voice", default="default", help="a voice id of the served model")
    parser.add_argument("--no-tools", action="store_true", help="for models that reject tools")
    parser.add_argument("--scenario", default="conversation", choices=sorted(SCENARIOS))
    parser.add_argument("--script", help="JSON step list instead of a built-in scenario")
    parser.add_argument("--out", default=str(HERE / "runs"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, steps in SCENARIOS.items():
            print(name, json.dumps(steps))
        return
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

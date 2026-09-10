#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "livekit-agents[openai,turn-detector,silero]>=1.6.10",
# ]
# ///
import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins.openai.realtime import RealtimeModel

logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = f"http://{os.environ['VLLM_OMNI_HOST']}:8000/v1"
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")

# "client": the agent runs its own VAD/turn detector and the realtime model
# is never asked to decide turns. Works with any model behind vLLM-Omni's
# OpenAI-compatible realtime endpoint, including ones with no VAD of their
# own, so it's the safe default.
# "semantic": turn-taking is left to the realtime model's own server-side
# VAD. Only works if the model behind vLLM-Omni implements it.
VAD_MODE = os.environ.get("VAD_MODE", "client")
if VAD_MODE not in ("client", "semantic"):
    raise ValueError(f"VAD_MODE must be 'client' or 'semantic', got {VAD_MODE!r}")

server = AgentServer(
    ws_url=LIVEKIT_URL,
    api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
)


class VoiceAssistant(Agent):
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, str | int]:
        """Return deterministic sample data for the demo tool."""
        return {
            "location": location,
            "weather": "clear skies",
            "temperature_f": 72,
        }

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely.",
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Omni serves one model, so an empty model lets the server select it.
    model = RealtimeModel(
        base_url=VLLM_BASE_URL,
        model="",
        api_key="unused",
    )

    if VAD_MODE == "client":
        # turn_handling's client-side TurnDetector means the server isn't
        # asked to decide turns, so turn_detection is left NOT_GIVEN on
        # RealtimeModel.
        turn_handling = TurnHandlingOptions(turn_detection=inference.TurnDetector())
    else:
        # No client-side turn detector, so the session falls back to the
        # realtime model's own (server-side) turn detection.
        turn_handling = TurnHandlingOptions()

    session = AgentSession(
        llm=model,
        turn_handling=turn_handling,
    )

    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info(
        "Voice assistant started (vad_mode=%s), connected to vLLM-Omni at %s",
        VAD_MODE,
        VLLM_BASE_URL,
    )


if __name__ == "__main__":
    cli.run_app(server)

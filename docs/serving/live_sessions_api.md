# OpenAI Live Sessions API

`WS /v1/live/sessions` serves the [OpenAI Live API](https://developers.openai.com/api/docs/guides/live)
(primary WebSocket transport) for speech-to-speech models. A Live session is a
timeline rather than a sequence of turns: audio in, audio out, timestamped
transcript fragments, and tool use delegated to a Responses backend, which on
this server is the served model itself.

Two kinds of models sit behind the same endpoint:

| `vad` | Models | Who decides when to speak |
| ------- | -------- | --------------------------- |
| `external` | Qwen3-Omni | A Silero VAD service detects end of user speech; the server runs one generation turn per utterance, paces the reply at 1x, and truncates it on barge-in. |
| `native` | MiniCPM-o 4.5, PersonaPlex | The model hears the input continuously (1 s units for MiniCPM-o, 80 ms frames for PersonaPlex) and decides to listen or speak. |

Each session is one vLLM resumable request that the server appends to; the
server keeps the conversation history and resubmits it when it must change
(interruption, left-trimming, a changed tool list).

## Deployment

A Live deployment runs three processes: the VAD service (external-VAD models
only), a Whisper ASR service, and vLLM-Omni with a `*_live.yaml` deploy
overlay.

```bash
# 1. VAD service (CPU; external-VAD models only)
python -m vllm_omni.entrypoints.live_vad_service --port 15151

# 2. ASR service (plain vLLM, not --omni)
vllm serve openai/whisper-large-v3-turbo --port 15152 --gpu-memory-utilization 0.08

# 3. The model
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8000 \
  --deploy-config vllm_omni/deploy/qwen3_omni_live.yaml
```

The endpoint is enabled only when both the model declares a
`live_session_config` (in its pipeline) and the deploy config carries a
`live_session_config` object. Otherwise the WebSocket is accepted, answered
with an `error`, and closed with code 1008. Live sessions need a single API
server process (`--api-server-count 1`, the default).

On every `session.start` the server checks `GET /health` on the services the
model uses. If a service is down at start or fails mid-session, the client
receives a generic `error` (`type: "server_error"`, `code:
"internal_server_error"`) and the socket closes; the failing URL is logged
server-side.

### Deploy config

```yaml
live_session_config:
  session_lifetime_s: 1800          # hard limit; fills expires_at
  external_vad:                     # vad: external only
    output_audio_delta_size_ms: 500 # pacer chunk size = playback-cursor precision
    vad_service_url: ws://localhost:15151
  native_vad:                       # vad: native only
    audio_transcription_interval_ms: 10000
  asr_service_url: http://localhost:15152
  asr_timeout_s: 10
```

Bundled overlays: `qwen3_omni_live.yaml`, `minicpmo_4_5_live.yaml`,
`personaplex_live.yaml`.

### VAD service

`vllm_omni.entrypoints.live_vad_service` wraps `SileroStreamingVAD` (ONNX, CPU,
one shared detector). `WS /v1/vad` takes binary 16 kHz PCM16 frames and answers
each with a JSON result (`speech_started`, `speech_stopped`, `speech_start_ms`,
`speech_end_ms`, `speech_probability`); a `{"type": "reset"}` text frame drops
endpointing state. Endpointing is configured on the service, not by clients:

| Flag | Default | Meaning |
| ------ | --------- | --------- |
| `--model-path` | pinned HF artifact | Silero VAD v6.2 ONNX file |
| `--threshold` | 0.5 | Speech probability threshold |
| `--prefix-padding-ms` | 300 | Audio kept before detected speech |
| `--silence-duration-ms` | 500 | Silence that ends an utterance |
| `--min-speech-duration-ms` | 96 | Shorter speech (backchannels) never starts a turn |
| `--host` / `--port` | 127.0.0.1 / 15151 | Listen address |

### ASR service

Any vLLM deployment of an `openai/whisper-*` checkpoint. The server posts
`verbose_json` requests; only segment-level timestamps are used. Whisper
supplies `session.input_transcript.delta` on every model and, on external-VAD
models, the timestamps of `session.output_transcript.delta` and the token
position for barge-in truncation.

## Protocol

Messages are JSON text frames validated against the `openai.types.live`
models. Audio is base64 raw audio in the session format with no container:
`audio/pcm` at 16000 or 24000 Hz (default 24000), `audio/pcmu`, or
`audio/pcma`. The same format applies to both directions.

### Client events

| Event | Behaviour |
| ------- | ----------- |
| `session.start` | Must be first. `model` must be the served model name. `audio.output.voice` is a model voice id (OpenAI voice names are rejected). Replies `session.started`, then `info` (`code: "capabilities"`) listing unsupported features. Invalid configuration: `error`, then close 1008. |
| `session.update` | Only `delegation.responses` may change. Changed `instructions`, `tools`, or `tool_choice` re-render the history at the next generation turn. Replies `session.updated`. |
| `session.input_audio.append` | Advances the session clock by the chunk's duration. Malformed audio: `error` `invalid_audio`. |
| `session.input_audio.mute` / `.unmute` | External VAD: input stops reaching the VAD and any speech in progress is dropped. Native VAD: the model is fed silence. Replies `.muted` / `.unmuted`. |
| `session.instructions.append` | `delegation_id` must be present and `null`. Added as developer context at the next generation turn. Replies `session.instructions.appended`. |
| `response.item.create` | `delegation.responses` sessions only. Accepts `function_call_output` for an outstanding call, or a user message. |
| `response.create` | `delegation.responses` sessions only. Runs the continuation after tool results. |
| `session.close` | Replies `session.usage.updated` and `session.closed` (`reason: "close_requested"`), then closes 1000. |
| `session.commentary.append` | Accepted and ignored (no reply). |
| `session.thinking.append` | `error` `unsupported_event`. |

Unknown events and events a model lists as unsupported get `error` with
`code: "unsupported_event"`. Events other than `session.start` before the
session starts get `session_not_started`. Non-null `store`, `client`, and
`delegation: {type: "client"}` are `unsupported_session_config`.

### Server events

| Event | Notes |
| ------- | ------- |
| `session.output_audio.delta` | External VAD: released at 1x in `output_audio_delta_size_ms` chunks. Native VAD: as the model produces it. |
| `session.output_transcript.delta` | The model's own text. External VAD: after the audio chunk it belongs to, with Whisper-aligned `start_ms`/`end_ms`. Native VAD: timestamped with the input window that produced it. |
| `session.input_transcript.delta` | Whisper transcript of each utterance (external) or each `audio_transcription_interval_ms` window (native). |
| `session.delegation.created`, `response.event` | Tool calls; see below. |
| `session.usage.updated` | Every 60 s of session time and at close. |
| `session.closed` | `close_requested` or `expired` (`expires_at` reached). |

### Tool calling

The served model is its own Responses backend: `delegation.responses.model`
must equal the served model (or be omitted), and `tools`, `tool_choice`
(`auto`/`none`), `instructions`, and `max_output_tokens` are rendered into the
prompt and sampling parameters. When the model emits a tool call, the server
sends `session.delegation.created` followed by `response.event` frames wrapping
`response.created`, `response.output_item.added`,
`response.function_call_arguments.delta`/`.done`, `response.output_item.done`,
and `response.completed`. The client answers with `response.item.create`
(`function_call_output`) per call and then `response.create`; the spoken
answer arrives as ordinary output audio and transcript.

Tools are supported on Qwen3-Omni (hermes format; `--tool-call-parser`
overrides it when `--enable-auto-tool-choice` is set). MiniCPM-o 4.5 and
PersonaPlex reject non-empty `delegation.responses.tools`; a
`delegation.responses` without tools is accepted everywhere.

## Barge-in (external VAD)

Generated audio is queued and released at real-time pace; the released
position is the playback cursor. When the VAD service reports speech while
output is still being generated or is queued, the server aborts the request,
discards unreleased audio, maps the cursor to an output token through the
Whisper alignment, truncates the assistant turn there, and resubmits the
history with the next user turn. Precision is bounded by
`output_audio_delta_size_ms`. Speech shorter than the VAD service's
`--min-speech-duration-ms` never interrupts.

## Model notes

- **Qwen3-Omni** (`vad: external`): voices `chelsie`, `ethan`, `aiden`. The
  newest 32 user audio turns are rendered as audio on a full re-render; older
  ones fall back to their transcripts. History is left-trimmed by whole turns
  when it would exceed `max_model_len`.
- **MiniCPM-o 4.5** (`vad: native`, 1 s units): voice `default`. The duplex
  listen/speak grammar is enforced by `MiniCPMODuplexLogitsProcessor`, which
  the Live overlay installs on stage 0. `instructions` and text `input` become
  the duplex system prompt.
- **PersonaPlex** (`vad: native`, 80 ms frames): voices are the bundled voice
  prompts (`NATF2`, ...); `instructions` is the persona. Output transcripts are
  not produced.
- **Nemotron VoiceChat** is not served on `/v1/live/sessions`: its thinker is
  frame-locked with no turn-based generation profile.

## Example

`examples/online_serving/live_sessions/livekit/` runs a LiveKit Agents voice
agent (`GPTLiveModel`) against this endpoint, plus a headless scripted driver
and a raw WebSocket probe.

## Metrics

`vllm_omni:live_active_sessions`, `vllm_omni:live_audio_seconds{direction}`,
`vllm_omni:live_pacer_unreleased_s`, `vllm_omni:live_asr_pending_requests`,
`vllm_omni:live_asr_latency_s`, `vllm_omni:live_first_audio_latency_s`,
`vllm_omni:live_interruptions`, and `vllm_omni:live_errors{reason}`, labelled
by `model_name` and `vad`.

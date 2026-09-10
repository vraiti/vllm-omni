# vLLM-Omni LiveKit demo

This directory contains a Compose version of the local LiveKit demo. It
builds the LiveKit server, the Next.js frontend, and the Python agent as
separate images. The vLLM-Omni server runs outside Compose and is configured
with `VLLM_OMNI_HOST`.

Every service runs with `network_mode: host`, so they share the host's
network namespace directly instead of a Compose bridge network: no
container-to-container DNS, no port publishing/remapping (each service binds
its configured port straight to the host), and `VLLM_OMNI_HOST` only needs
to be an address the host itself can reach -- no `host.docker.internal`
plumbing required.

Start vLLM-Omni with port `8000` bound to an address reachable from the
container (e.g. `--host 0.0.0.0`), then run:

```bash
cd examples/online_serving/openai-fullduplex/livekit
VLLM_OMNI_HOST=localhost podman compose up --build
```

Or use the `compose-livekit` wrapper in `vllm-omni-aux/utils/`, which
resolves `VLLM_OMNI_HOST` from an ssh alias via `ip-of` and runs this from
anywhere in the checkout:

```bash
compose-livekit [client|semantic] [ssh-alias]   # e.g. compose-livekit semantic dev
```

Open [http://localhost:3000](http://localhost:3000). The LiveKit server is
available at `ws://localhost:7880`; its development credentials are `devkey`
and `secret`.

For vLLM-Omni running on a different machine, set both the vLLM address and
the address LiveKit advertises to WebRTC clients:

```bash
VLLM_OMNI_HOST=192.168.1.20 \
LIVEKIT_NODE_IP=192.168.1.10 \
LIVEKIT_PUBLIC_URL=ws://192.168.1.10:7880 \
podman compose up --build
```

Use `podman compose up --build --abort-on-container-exit` when the Compose
process should stop the remaining services after one component exits, matching
the behavior of `run-livekit-stack.sh`.

## Turn detection

The agent decides who's speaking either client-side (its own VAD/turn
detector) or by leaving it to the realtime model's server-side VAD, set via
`VAD_MODE`:

```bash
VAD_MODE=semantic VLLM_OMNI_HOST=localhost podman compose up --build
```

- `client` (default): works with any model behind vLLM-Omni, including ones
  with no VAD of their own.
- `semantic`: server-side VAD; only works if the model behind vLLM-Omni
  implements it.

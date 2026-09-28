"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
try:
    from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402
    from vllm_omni.transformers_utils import parsers as _parsers  # noqa: F401, E402
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    pass

from .config import OmniModelConfig


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        from .entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from .entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]


def _install_call_trace() -> None:
    """Debug call tracing, enabled by RUN_REMOTE_CALL_TRACE=1.

    Every call of a named vllm_omni function (qualified name without '<' or
    '>') appends "(pid, thread, coro, depth) <module>.<qualname>" to
    /tmp/logs/call-trace/call-trace_<pid>_<thread>_<coro>.log, created
    (overwriting) on first use. thread is the OS thread id; coro numbers the
    outermost coroutine on the stack (for asyncio, the task) per process, 0
    meaning no coroutine; depth is the number of Python frames on the
    thread's stack, the called function included. Generator/coroutine frames
    are logged on their first start only. Installed on every thread; each
    process that imports vllm_omni traces itself.
    """
    import dis
    import inspect
    import os
    import sys
    import threading
    import weakref

    directory = "/tmp/logs/call-trace"
    os.makedirs(directory, exist_ok=True)
    prefix = __name__
    resume = dis.opmap["RESUME"]
    generator_flags = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR

    names: dict = {}  # code -> "<module>.<qualname>" | None
    starts: dict = {}  # generator code -> byte offset of its first RESUME
    coro_numbers: dict = {}  # id(root coroutine) -> number
    coro_refs: dict = {}  # id(root coroutine) -> weakref that forgets it
    files: dict = {}  # (coro number, thread) -> line-buffered file
    state = {"pid": os.getpid(), "next": 1, "failed": False}

    def qualified_name(frame):
        code = frame.f_code
        try:
            return names[code]
        except KeyError:
            pass
        module = frame.f_globals.get("__name__")
        qualname = code.co_qualname
        name = None
        if (
            code.co_flags & inspect.CO_OPTIMIZED  # a function, not a class body or module
            and isinstance(module, str)
            and (module == prefix or module.startswith(prefix + "."))
            and "<" not in qualname
            and ">" not in qualname
        ):
            name = f"{module}.{qualname}"
        names[code] = name
        return name

    def first_start(frame):
        # Resumptions and throws (including a finalizer's close()) also arrive
        # as "call" events; only the first start is at the first RESUME.
        code = frame.f_code
        offset = starts.get(code)
        if offset is None:
            raw = code.co_code
            offset = next((i for i in range(0, len(raw), 2) if raw[i] == resume), -2)
            starts[code] = offset
        return frame.f_lasti == offset

    def forget(key, _ref):
        number = coro_numbers.pop(key, None)
        coro_refs.pop(key, None)
        for file_key in [k for k in files if k[0] == number]:
            files.pop(file_key).close()

    def root_and_depth(frame):
        root = None
        depth = 0
        while frame is not None:
            depth += 1
            generator = getattr(frame, "f_generator", None)  # 3.14+
            if inspect.iscoroutine(generator):
                root = generator
            frame = frame.f_back
        return root, depth

    def log(frame):
        name = qualified_name(frame)
        if name is None:
            return
        pid = os.getpid()
        if pid != state["pid"]:  # a fork child starts over under its own pid
            state.update(pid=pid, next=1)
            files.clear()
            coro_numbers.clear()
            coro_refs.clear()
        if frame.f_code.co_flags & generator_flags and not first_start(frame):
            return
        coro = 0
        root, depth = root_and_depth(frame)
        if root is not None:
            key = id(root)
            coro = coro_numbers.get(key)
            if coro is None:
                coro = coro_numbers[key] = state["next"]
                state["next"] += 1
                coro_refs[key] = weakref.ref(root, lambda ref, key=key: forget(key, ref))
        thread = threading.get_native_id()
        file = files.get((coro, thread))
        if file is None:
            path = f"{directory}/call-trace_{pid}_{thread}_{coro}.log"
            file = files[(coro, thread)] = open(path, "w", buffering=1)
            root_name = getattr(root, "__qualname__", "?") if root is not None else "(no coroutine)"
            file.write(f"# pid={pid} thread={thread} coro={coro} root={root_name}\n")
        file.write(f"({pid}, {thread}, {coro}, {depth}) {name}\n")

    def trace(frame, event, arg):
        if event == "call":
            try:
                log(frame)
            except Exception:
                # A raising trace function would be removed and could break
                # the traced code; report once and keep going.
                if not state["failed"]:
                    state["failed"] = True
                    import traceback

                    traceback.print_exc()
        return None  # no local tracing: no line/return events

    if hasattr(threading, "settrace_all_threads"):  # 3.12+
        threading.settrace_all_threads(trace)
    else:
        threading.settrace(trace)
        sys.settrace(trace)


import os as _os  # noqa: E402

if _os.environ.get("RUN_REMOTE_CALL_TRACE") == "1":
    _install_call_trace()

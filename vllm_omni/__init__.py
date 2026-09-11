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

import asyncio
import inspect
import os
import sys
import threading

_TRACE_PACKAGE_ROOT = os.path.realpath(os.path.dirname(__file__))
_TRACE_LOG_DIRECTORY = "/tmp/logs/traces"
_TRACE_LOGS = {}
_TRACE_LOG_DISABLED = False


def _trace_is_tensor(value):
    try:
        return any(cls.__module__ == "torch" and cls.__name__ == "Tensor" for cls in type(value).__mro__)
    except Exception:
        return False


def _trace_tensor_repr(value):
    try:
        return f"<torch.Tensor size={tuple(value.size())}>"
    except Exception:
        try:
            return f"<torch.Tensor size={tuple(value.shape)}>"
        except Exception:
            return "<torch.Tensor size=<unavailable>>"


def _trace_repr(value, seen=None):
    if _trace_is_tensor(value):
        return _trace_tensor_repr(value)

    if seen is None:
        seen = set()

    if isinstance(value, dict):
        value_id = id(value)
        if value_id in seen:
            return "<recursive>"
        seen.add(value_id)
        try:
            items = [f"{_trace_repr(key, seen)}: {_trace_repr(item, seen)}" for key, item in value.items()]
            return "{" + ", ".join(items) + "}"
        finally:
            seen.remove(value_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        value_id = id(value)
        if value_id in seen:
            return "<recursive>"
        seen.add(value_id)
        try:
            items = [_trace_repr(item, seen) for item in value]
            if isinstance(value, list):
                return "[" + ", ".join(items) + "]"
            if isinstance(value, tuple):
                suffix = "," if len(items) == 1 else ""
                return "(" + ", ".join(items) + suffix + ")"
            if isinstance(value, frozenset):
                return "frozenset({" + ", ".join(items) + "})"
            return "{" + ", ".join(items) + "}"
        finally:
            seen.remove(value_id)

    try:
        return repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging fallback
        return f"<unrepresentable {type(value).__name__}: {exc}>"


def _trace_is_scoped_frame(frame):
    function_name = frame.f_code.co_name
    if function_name.startswith("<") and function_name.endswith(">"):
        return False

    filename = os.path.realpath(frame.f_code.co_filename)
    return filename == _TRACE_PACKAGE_ROOT or filename.startswith(_TRACE_PACKAGE_ROOT + os.sep)


def _trace_scope_depth(frame):
    depth = 0
    current_frame = frame
    while current_frame is not None:
        if _trace_is_scoped_frame(current_frame):
            depth += 1
        current_frame = current_frame.f_back
    return depth


def _trace_log_path():
    thread_id = threading.get_ident()
    process_id = os.getpid()
    path = f"{thread_id}-{process_id}"

    try:
        task = asyncio.current_task()
    except (AttributeError, RuntimeError):
        task = None

    if task is not None:
        try:
            coroutine_id = id(task.get_coro())
        except Exception:
            coroutine_id = id(task)
        path += f"-co{coroutine_id}"

    return os.path.join(_TRACE_LOG_DIRECTORY, path + ".log")


def _trace_call(frame, event, _arg):
    global _TRACE_LOG_DISABLED

    if event != "call":
        return _trace_call

    if not _trace_is_scoped_frame(frame):
        return _trace_call

    if _TRACE_LOG_DISABLED:
        return _trace_call

    try:
        arg_info = inspect.getargvalues(frame)
        local_values = frame.f_locals
        arguments = [f"{name}={_trace_repr(local_values.get(name))}" for name in arg_info.args]
        if arg_info.varargs is not None:
            name = arg_info.varargs
            arguments.append(f"*{name}={_trace_repr(local_values.get(name))}")
        if arg_info.keywords is not None:
            name = arg_info.keywords
            arguments.append(f"**{name}={_trace_repr(local_values.get(name))}")

        filename = os.path.realpath(frame.f_code.co_filename)
        function_name = frame.f_code.co_name
        qualified_name = getattr(frame.f_code, "co_qualname", function_name)
        argument_text = "\n\t".join(arguments)
        message = f"({_trace_scope_depth(frame)}) {filename}:{qualified_name}({argument_text})"

        log_path = _trace_log_path()
        log = _TRACE_LOGS.get(log_path)
        if log is None:
            os.makedirs(_TRACE_LOG_DIRECTORY, exist_ok=True)
            log = open(log_path, "a", encoding="utf-8", buffering=1)
            _TRACE_LOGS[log_path] = log
        print(message, file=log, flush=True)
    except Exception:  # pragma: no cover - tracing must not affect execution
        _TRACE_LOG_DISABLED = True
        for log in _TRACE_LOGS.values():
            try:
                log.close()
            except Exception:
                pass
        _TRACE_LOGS.clear()

    return _trace_call


sys.settrace(_trace_call)

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: E402, F401

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

from .config import OmniModelConfig  # noqa: E402


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

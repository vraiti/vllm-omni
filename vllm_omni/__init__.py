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

import inspect
import os
import sys

_TRACE_PACKAGE_ROOT = os.path.realpath(os.path.dirname(__file__))
_TRACE_LOG_PATH = "/tmp/logs/trace.log"
_TRACE_LOG = None
_TRACE_LOG_DISABLED = False


def _trace_repr(value):
    try:
        return repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging fallback
        return f"<unrepresentable {type(value).__name__}: {exc}>"


def _trace_call(frame, event, _arg):
    global _TRACE_LOG, _TRACE_LOG_DISABLED

    if event != "call":
        return _trace_call

    code = frame.f_code
    function_name = code.co_name
    if function_name.startswith("<") and function_name.endswith(">"):
        return _trace_call

    filename = os.path.realpath(code.co_filename)
    if not (filename == _TRACE_PACKAGE_ROOT or filename.startswith(_TRACE_PACKAGE_ROOT + os.sep)):
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

        module_name = frame.f_globals.get("__name__", "<unknown>")
        qualified_name = getattr(code, "co_qualname", function_name)
        message = f"{filename}:{frame.f_lineno} {module_name}.{qualified_name}({', '.join(arguments)})"

        if _TRACE_LOG is None:
            os.makedirs(os.path.dirname(_TRACE_LOG_PATH), exist_ok=True)
            _TRACE_LOG = open(_TRACE_LOG_PATH, "a", encoding="utf-8", buffering=1)
        print(message, file=_TRACE_LOG, flush=True)
    except Exception:  # pragma: no cover - tracing must not affect execution
        _TRACE_LOG_DISABLED = True
        if _TRACE_LOG is not None:
            try:
                _TRACE_LOG.close()
            except Exception:
                pass
            _TRACE_LOG = None

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

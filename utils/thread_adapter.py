
import threading
from contextlib import contextmanager

_thread_local = threading.local()


@contextmanager
def adapter_context(adapter_name: str):
    prev = getattr(_thread_local, "adapter_override", None)
    _thread_local.adapter_override = adapter_name
    try:
        yield
    finally:
        _thread_local.adapter_override = prev


def get_thread_adapter_override():
    return getattr(_thread_local, "adapter_override", None)


def install_thread_local_adapter_patch():
    from peft.tuners.tuners_utils import BaseTunerLayer

    _original_getter = BaseTunerLayer.active_adapter.fget

    def _thread_aware_active_adapter(self):
        override = get_thread_adapter_override()
        if override is not None:
            return override
        return _original_getter(self)

    BaseTunerLayer.active_adapter = property(_thread_aware_active_adapter)

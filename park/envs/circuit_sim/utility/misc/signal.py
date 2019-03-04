import contextlib
import signal

__all__ = ['handle_signal']


@contextlib.contextmanager
def handle_signal(signum, handler):
    original = signal.getsignal(signum)
    try:
        signal.signal(signum, handler)
        yield
    finally:
        signal.signal(signum, original)

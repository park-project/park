import functools
import threading

__all__ = ['synchronized']


# Refer: https://stackoverflow.com/questions/29158282/how-to-create-a-synchronized-function-across-all-instances
def synchronized(wrapped):
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)

    return wrapper

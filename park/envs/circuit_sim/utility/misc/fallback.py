import functools

__all__ = ['fallback_wrap', 'fallback_call', 'fallback_get_attr', 'fallback_call_method']


def fallback_wrap(fallback=None, fallback_callable=None):
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return fallback_call(fn, args, kwargs, fallback, fallback_callable)

        return wrapped

    return wrapper


def fallback_call(fn, args=None, kwargs=None, fallback=None, fallback_callable=None):
    args = args or ()
    kwargs = kwargs or {}
    try:
        return fn(*args, **kwargs)
    except:
        if fallback_callable is None:
            fallback_callable = callable(fallback)
        return fallback(*args, **kwargs) if fallback_callable else fallback


def fallback_get_attr(obj, key, fallback=None, set_on_failure=False):
    try:
        return getattr(obj, key)
    except:
        if set_on_failure:
            setattr(obj, key, fallback)
        return fallback


def fallback_call_method(obj, key, args=None, kwargs=None, fallback=None,
                         fallback_callable=None):
    try:
        return getattr(obj, key)(*args, **kwargs)
    except:
        if fallback_callable is None:
            fallback_callable = callable(fallback)
        return fallback(*args, **kwargs) if fallback_callable else fallback

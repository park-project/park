import os

from utility.misc.registry import WrappedRegistry

__all__ = ['get_environ_registry', 'set_environ']


def get_environ_registry() -> WrappedRegistry:
    if not hasattr(get_environ_registry, '__registry__'):
        setattr(get_environ_registry, '__registry__', WrappedRegistry(os.environ))
    return getattr(get_environ_registry, '__registry__')


def set_environ(key, value):
    get_environ_registry().register(key, value)

from utility.misc.meta import Singleton
from utility.misc.registry import DefaultRegistry

__all__ = ['Namer']


@Singleton
class Namer(object):
    def __init__(self):
        self._registry = DefaultRegistry()

    def get_name(self, klass, name):
        assert isinstance(klass, type) and isinstance(name, str)
        record = self._registry.lookup(klass)
        record.setdefault(name, -1)
        record[name] += 1
        if record[name] == 0:
            return name
        return f'{name}_{record[name]}'

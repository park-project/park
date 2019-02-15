import collections

__all__ = ['AttrDict', 'flatten_by_meta', 'flatten']


class AttrDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(item)
        return self[item]

    def __setattr__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)

    @classmethod
    def nested_attr(cls, data: dict):
        results = {k: cls.nested_attr(v) if isinstance(v, dict) else v for k, v in data.items()}
        return AttrDict(**results)


def flatten(container):
    if isinstance(container, dict) or isinstance(container, collections.Iterable):
        container_values = container.values() if isinstance(container, dict) else list(container)
        results = []
        for i in container_values:
            results.extend(flatten(i))
        return results
    else:
        return [container]


def flatten_by_meta(data, meta):
    results = []
    for key, space in meta.items():
        if isinstance(space, dict):
            results.extend(flatten_by_meta(data[key], space))
        else:
            results.append(data[key])
    return results

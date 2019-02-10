__all__ = ['AttrDict']


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

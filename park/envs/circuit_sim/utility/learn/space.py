import abc
import collections
import functools

import numpy as np

__all__ = ['DatSpace', 'ObjSpace', 'Box', 'Discrete', 'SCALAR_SPACE', 'SpaceResolver']


class ObjSpace(object):
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class DatSpace(ObjSpace, metaclass=abc.ABCMeta):
    def __init__(self, shape, dtype):
        super().__init__(dtype)
        self._dtype = dtype
        self._shape = self._convert_to_shape(shape)

    @property
    def dtype(self):
        return self._dtype

    @staticmethod
    def _convert_to_shape(shape):
        return tuple(np.atleast_1d(np.asarray(shape, dtype=np.int)).tolist())

    @property
    def shape(self):
        return self._shape

    def zeros(self, size=None, order=None):
        if size is not None:
            return np.zeros((size,) + self.shape, self.dtype, order)
        else:
            return np.zeros(self.shape, self.dtype, order)

    def ones(self, size=None, order=None):
        if size is not None:
            return np.ones((size,) + self.shape, self.dtype, order)
        else:
            return np.ones(self.shape, self.dtype, order)

    def __str__(self):
        return f'{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})'

    __repr__ = __str__


class Box(DatSpace):
    def __init__(self, shape, dtype=None, bound=None):
        super().__init__(shape, dtype or 'float64')

        if bound is None:
            bound = -np.inf, np.inf

        self._min_bound, self._max_bound = self._formalize_bound(bound, self._shape, self._dtype)

    @staticmethod
    def _formalize_bound(bound, shape=None, dtype=None):
        min_bound, max_bound = bound
        min_bound = np.asarray(min_bound, dtype=dtype)
        max_bound = np.asarray(max_bound, dtype=dtype)

        if shape is not None:
            if len(min_bound.shape) == 0:
                min_bound = np.ones(shape) * min_bound
            if len(max_bound.shape) == 0:
                max_bound = np.ones(shape) * max_bound

            assert min_bound.shape == shape
            assert max_bound.shape == shape

        if dtype is not None:
            min_bound = min_bound.astype(dtype)
            max_bound = max_bound.astype(dtype)

        return min_bound, max_bound

    @classmethod
    def from_bound(cls, bound, dtype=None):
        min_bound, max_bound = cls._formalize_bound(bound, dtype=dtype)
        return cls(min_bound.shape, min_bound.dtype, (min_bound, max_bound))

    @property
    def min_bound(self):
        return self._min_bound

    @property
    def max_bound(self):
        return self._max_bound

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def bound(self):
        return self.min_bound, self.max_bound

    def normalize(self, value, target_bound=(-1., 1.)):
        target_min_bound, target_max_bound = self._formalize_bound(target_bound, self._shape, self._dtype)
        value = (value - self.min_bound) / (self.max_bound - self.min_bound)
        value = value * (target_max_bound - target_min_bound) + target_min_bound
        return value

    def denormalize(self, value, source_bound=(-1., 1.)):
        source_min_bound, source_max_bound = self._formalize_bound(source_bound, self._shape, self._dtype)
        value = (value - source_min_bound) / (source_max_bound - source_min_bound)
        value = value * (self.max_bound - self.min_bound) + self.min_bound
        return value

    def sample(self, bound=None, state: np.random.RandomState = None):
        bound = bound or self.bound
        min_bound, max_bound = self._formalize_bound(bound, shape=self._shape, dtype=self._dtype)
        assert np.isfinite(min_bound).all() and np.isfinite(max_bound).all(), "Bound must be finite to sample from."
        state = state or np.random
        return state.rand(*self._shape) * (max_bound - min_bound) + min_bound


class Discrete(DatSpace):
    def __init__(self, options, dtype=None):
        if isinstance(options, int):
            self._options = options
        else:
            self._options = list(options)
        super().__init__((), dtype or 'int64')

    @property
    def options(self):
        return self._options


SCALAR_SPACE = Box(())


class SpaceResolver(object):
    def __init__(self, space):
        assert isinstance(space, (ObjSpace, dict, collections.Iterable)), f"Cannot resolve {space.__class__.__name__}"
        self._space = space
        self._names = SpaceResolver._resolve(self._space, None)

    @property
    def space(self):
        return self._space

    def name_value(self, name):
        return self._names or name

    def build_map(self, keys, values):
        results = {}
        SpaceResolver._resolve(self._space, results.setdefault, keys, values)
        return results

    def resolve(self, value, *args, scope=None):
        return self._resolve(self._space, value, *args, scope=scope)

    @classmethod
    def _zeros(cls, space, size=None, order=None):
        assert isinstance(space, DatSpace), f"Cannot produce zeros from {space.__class__.__name__}"
        return space.zeros(size, order)

    def zeros(self, size=None, order=None):
        return SpaceResolver._resolve(self._space, functools.partial(self._zeros, size=size, order=order), self._space)

    @classmethod
    def _ones(cls, space, size=None, order=None):
        assert isinstance(space, DatSpace), f"Cannot produce ones from {space.__class__.__name__}"
        return space.ones(size, order)

    def ones(self, size=None, order=None):
        return SpaceResolver._resolve(self._space, functools.partial(self._ones, size=size, order=order), self._space)

    @classmethod
    def _resolve(cls, space, value, *args, scope=None, default=None):
        if isinstance(space, dict) or isinstance(space, collections.Iterable):
            space_with_key = space.items() if isinstance(space, dict) else enumerate(space)
            results = {}
            for k, v in space_with_key:
                new_args = [arg.get(k, default) for arg in args]
                result = cls._resolve(v, value, *new_args, scope=k)
                results.setdefault(k, result)
            return space.__class__(**results) if isinstance(space, dict) else list(results.values())
        else:
            if not value:
                return scope
            if callable(value):
                return value(*args)
            return value

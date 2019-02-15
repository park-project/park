import abc
import concurrent.futures
import functools
import os

from park import logger
from park.envs.circuit_sim.utils import open_tmp_path, AttrDict, loads_pickle, dumps_pickle, make_pool, RobustClient

__all__ = ['Context', 'RemoteContext', 'AsyncLocalContext', 'LocalContext']


class Context(object, metaclass=abc.ABCMeta):
    __current_context = []

    def __init__(self, debug=False):
        self._debug = debug
        self.__opened = False

    @staticmethod
    def _evaluate(path, circuit, values, debug):
        with open_tmp_path(os.path.join(path, circuit.__class__.__name__), 'timepid', debug) as path:
            return circuit.run(path, AttrDict(**values))

    @abc.abstractmethod
    def evaluate(self, circuit, values, debug=None):
        pass

    @abc.abstractmethod
    def evaluate_batch(self, circuit, values, debug=None):
        pass

    @classmethod
    def current_context(cls):
        return cls.__current_context[-1]

    @property
    def opened(self):
        return self.__opened

    def __enter__(self):
        assert not self.__opened, "Context cannot be reopened."
        self.__current_context.append(self)
        self.__opened = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__opened = False
        self.__current_context.pop()

    def __repr__(self):
        return self.__str__()


class LocalContext(Context):
    def __init__(self, path, debug=False):
        super().__init__(debug)
        self._path = path
        self._pool = None

    @property
    def path(self):
        return self._path

    def __enter__(self):
        super(LocalContext, self).__enter__()
        self._pool = make_pool('process', os.cpu_count())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
        self._pool = None
        super(LocalContext, self).__exit__(exc_type, exc_val, exc_tb)

    def evaluate(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._evaluate(self._path, circuit, values, debug)

    def evaluate_batch(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._pool.map(functools.partial(self._evaluate, self._path, circuit, debug=debug), values)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return f'{self.__class__.__name__}(path={self._path})'


class AsyncLocalContext(LocalContext):
    def evaluate(self, circuit, values, debug=None):
        future = concurrent.futures.Future()
        debug = self._debug if debug is None else debug
        self._pool.apply_async(self._evaluate, (self._path, circuit, values, debug),
                               callback=future.set_result, error_callback=future.set_exception)
        return future

    def evaluate_batch(self, circuit, values, debug=None):
        future = concurrent.futures.Future()
        debug = self._debug if debug is None else debug
        func = functools.partial(self._evaluate, self._path, circuit, debug=debug)
        self._pool.map_async(func, values, callback=future.set_result, error_callback=future.set_exception)
        return future


class RemoteContext(Context):
    def __init__(self, host, port, debug=False):
        super().__init__(debug)
        self._host = host
        self._port = port
        self._client = RobustClient()

    def __enter__(self):
        super(RemoteContext, self).__enter__()
        self._client.initialize()
        self._client.connect('tcp', self._host, self._port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.finalize()
        super(RemoteContext, self).__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _encode(circuit, method, values, debug):
        name = circuit.__class__.__name__.encode('utf-8')
        method = str(method).encode('utf-8')
        values = dumps_pickle(values)
        debug = str(int(debug)).encode('utf-8')
        return name, method, values, debug

    def _request(self, name, method, values, debug):
        try:
            result, = self._client.request(name, method, values, debug)
            return loads_pickle(result)
        except:
            logger.exception("Exception occurred at remote server.")
            return None

    def evaluate(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._request(*self._encode(circuit, 'simulate', values, debug))

    def evaluate_batch(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._request(*self._encode(circuit, 'simulate_batch', values, debug))

    def __getstate__(self):
        raise NotImplementedError("Cannot pickle RemoteContext object.")

    def __setstate__(self, state):
        raise NotImplementedError("Cannot pickle RemoteContext object.")

    def __str__(self):
        return f'{self.__class__.__name__}(host={self._host}, port={self._port})'

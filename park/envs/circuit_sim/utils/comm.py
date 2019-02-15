import abc
import asyncio
import concurrent.futures
import contextlib
import functools

import zmq
import zmq.asyncio

from park import logger
from park.envs.circuit_sim.utils import graceful_execute, make_pool

__all__ = ['_AbstractNode', 'socket_bind', 'socket_unbind', 'socket_connect', 'socket_disconnect',
           'RobustClient', 'RobustServer']


class _AbstractNode(object, metaclass=abc.ABCMeta):
    def __init__(self):
        self._context = None
        self.__internal_context = None

    def initialize(self, context: zmq.Context = None):
        if self.__internal_context:
            raise ValueError("Node has already been activated.")
        if context:
            self.__internal_context = False
            return context
        self.__internal_context = True
        self._context = zmq.Context().instance()

    def finalize(self):
        if self.__internal_context:
            self._context.term()
        self._context = None
        self.__internal_context = None

    @property
    def context(self) -> zmq.Context:
        return self._context

    @contextlib.contextmanager
    def on_context(self, context) -> '_AbstractNode':
        try:
            self.initialize(context)
            yield self
        finally:
            self.finalize()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def socket_bind(socket: zmq.Socket, protocol, interface=None, port=None):
    if protocol == 'tcp':
        assert interface is None
        if port is None:
            port = socket.bind_to_random_port('tcp://0.0.0.0')
        else:
            socket.bind(f'tcp://0.0.0.0:{port}')
        return port
    else:
        assert interface is not None
        socket.bind(f'tcp://{interface}')


def socket_unbind(socket: zmq.Socket, protocol, interface=None, port=None):
    if protocol == 'tcp':
        assert port is not None and interface is None
        socket.unbind(f'tcp://0.0.0.0:{port}')
    else:
        assert interface is not None
        socket.unbind(f'tcp://{interface}')


def socket_connect(socket: zmq.Socket, protocol, interface, port=None):
    if protocol == 'tcp':
        assert port is not None
        socket.connect(f'{protocol}://{interface}:{port}')
    else:
        assert port is None
        socket.connect(f'{protocol}://{interface}')


def socket_disconnect(socket: zmq.Socket, protocol, interface, port=None):
    if protocol == 'tcp':
        assert port is not None
        socket.disconnect(f'{protocol}://{interface}:{port}')
    else:
        assert port is None
        socket.disconnect(f'{protocol}://{interface}')


class RobustClient(_AbstractNode):
    def __init__(self):
        super().__init__()
        self._socket = None
        self._counter = None

    def initialize(self, context: zmq.Context = None):
        super(RobustClient, self).initialize(context)
        assert isinstance(self._context, zmq.Context)
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.RCVHWM, 0)
        self._counter = 0

    def finalize(self):
        assert isinstance(self._socket, zmq.Socket)
        self._socket.close()
        self._socket = None
        self._counter = None
        super(RobustClient, self).finalize()

    def connect(self, protocol, interface, port=None):
        socket_connect(self._socket, protocol, interface, port)

    def request(self, *args, timeout=1):
        result, = tuple(self.irequest_unordered([args], timeout))
        return result

    def irequest_unordered(self, args_batch, timeout=1):
        tasks = {}
        for index, args in enumerate(args_batch):
            self._counter += 1
            tasks[str(self._counter).encode('utf-8')] = args
        while tasks:
            for index, args in tasks.items():
                self._socket.send_multipart((str(timeout).encode('utf-8'), index) + args, copy=False)

            while tasks:
                try:
                    flag, *payload = self._socket.recv_multipart(zmq.DONTWAIT, copy=True)
                except zmq.ZMQError:
                    has_events = bool(self._socket.poll(timeout * 1000))
                    if not has_events:
                        logger.info('Resend heartbeat signal because the server seems dead.')
                        self._socket.send_multipart([str(timeout).encode('utf-8'), b'h'], copy=False)
                        continue
                    flag, *payload = self._socket.recv_multipart(zmq.DONTWAIT, copy=True)
                if flag == b'h':
                    logger.debug(f'Received server response for heartbeat.')
                elif flag == b'e':
                    message, = payload
                    message = message.decode('utf-8')
                    raise RuntimeError(f'Error occurs when execute request on server: "{message}"')
                elif flag == b'r':
                    logger.debug(f'Resend the message required by the server')
                    break
                elif flag not in tasks:
                    logger.warning('Receive outdated index result from the server')
                else:
                    tasks.pop(flag)
                    yield payload


class RobustServer(_AbstractNode):
    def __init__(self, mode='thread', workers=None):
        super().__init__()
        self._socket = None
        self._workers = workers
        self._mode = mode
        self._pool = None
        self._ongoing_tasks = None

    def bind(self, protocol, interface=None, port=None):
        return socket_bind(self._socket, protocol, interface, port)

    def initialize(self, context: zmq.Context = None):
        super(RobustServer, self).initialize(context)
        assert isinstance(self._context, zmq.Context)
        self._socket = zmq.asyncio.Socket.from_socket(self._context.socket(zmq.ROUTER))
        self._socket.setsockopt(zmq.RCVHWM, 0)
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._pool = make_pool(self._mode, self._workers)
        self._ongoing_tasks = dict()

    def finalize(self):
        self._ongoing_tasks = None
        self._pool.terminate()
        self._pool = None
        assert isinstance(self._socket, zmq.Socket)
        self._socket.close()
        self._socket = None
        super(RobustServer, self).finalize()

    def _task_callback(self, identity, index, future: asyncio.Future):
        exception = future.exception()
        self._ongoing_tasks[identity].remove(index)
        if not self._ongoing_tasks[identity]:
            self._ongoing_tasks.pop(identity)
        if exception:
            logger.exception(f'An exception occurred when handling task.', exc_info=exception)
            coroutine = self._socket.send_multipart([identity, b'e', str(exception).encode('utf-8')], copy=False)
        else:
            result = future.result()
            if result is None:
                result = []
            elif isinstance(result, bytes):
                result = [result]
            elif isinstance(result, str):
                result = [result.encode('utf-8')]
            elif isinstance(result, tuple):
                result = list(result)
            else:
                raise ValueError('Result cannot be of type {!r}'.format(result))
            coroutine = self._socket.send_multipart([identity, index] + result, flags=zmq.DONTWAIT, copy=False)
        asyncio.ensure_future(coroutine)

    async def _heartbeat_after(self, identity, interval):
        while identity in self._ongoing_tasks:
            await asyncio.sleep(interval)
            await self._socket.send_multipart([identity, b'h'], flags=zmq.DONTWAIT, copy=False)

    async def _handler(self, identity, payload, callback):
        timeout, index, *payload = payload

        request_resend = False
        if identity not in self._ongoing_tasks:
            _ = asyncio.ensure_future(self._heartbeat_after(identity, float(timeout) / 2))
            self._ongoing_tasks[identity] = set()
            request_resend = True

        if index == b'h':
            if request_resend:
                await self._socket.send_multipart([identity, b'r'], flags=zmq.DONTWAIT, copy=False)
            else:
                await self._socket.send_multipart([identity, b'h'], flags=zmq.DONTWAIT, copy=False)
            return

        if index in self._ongoing_tasks[identity]:
            logger.info('Duplicated index found from the client')
            return
        self._ongoing_tasks[identity].add(index)

        if asyncio.iscoroutinefunction(callback):
            future = asyncio.ensure_future(callback(*payload))
        else:
            future = concurrent.futures.Future()
            self._pool.apply_async(callback, payload, callback=future.set_result,
                                   error_callback=future.set_exception)
            future = asyncio.wrap_future(future)
        future.add_done_callback(functools.partial(self._task_callback, identity, index))

    async def _mainloop(self, callback):
        assert isinstance(self._socket, zmq.asyncio.Socket)
        while True:
            identity, *payload = await self._socket.recv_multipart(copy=True)
            try:
                await self._handler(identity, payload, callback)
            except KeyboardInterrupt:
                raise
            except Exception as exception:
                logger.exception(f'An exception occurred when handling message.')
                await self._socket.send_multipart([identity, b'e', str(exception).encode('utf-8')],
                                                  flags=zmq.DONTWAIT, copy=False)

    async def _cancelable_mainloop(self, callback):
        try:
            await self._mainloop(callback)
        except asyncio.CancelledError:
            pass

    def mainloop(self, callback):
        graceful_execute(self._cancelable_mainloop(callback))

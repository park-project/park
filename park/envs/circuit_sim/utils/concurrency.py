import multiprocessing
import multiprocessing.pool
import os
import asyncio

__all__ = ['make_pool', 'graceful_shutdown', 'graceful_execute']


def make_pool(mode, workers, propagate_process_signal=False) -> multiprocessing.pool.Pool:
    assert mode in ('thread', 'process'), 'mode can only be thread or process'
    if mode == 'thread':
        return multiprocessing.pool.ThreadPool(workers)
    else:
        return multiprocessing.Pool(workers, initializer=None if propagate_process_signal else os.setpgrp)


def graceful_shutdown(loop: asyncio.AbstractEventLoop = None):
    tasks = asyncio.gather(*asyncio.Task.all_tasks(loop=loop), loop=loop, return_exceptions=True)
    tasks.add_done_callback(lambda t: loop.stop())
    tasks.cancel()

    while not tasks.done() and not loop.is_closed():
        loop.run_forever()
    tasks.exception()


def graceful_execute(coroutine, loop=None):
    loop = loop or asyncio.get_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    except KeyboardInterrupt:
        pass
    finally:
        graceful_shutdown(loop)

import time

__all__ = ['TimeMeter']


class TimeMeter(object):
    def __init__(self):
        self._stamp = time.time()
        self._last_stamp = None
        self._start_stamp = self._stamp
        self._interval = None

    def tick(self):
        self._last_stamp = self._stamp
        self._stamp = time.time()

    @property
    def elapsed(self):
        return self._stamp - self._start_stamp

    @property
    def interval(self):
        return self._stamp - self._last_stamp

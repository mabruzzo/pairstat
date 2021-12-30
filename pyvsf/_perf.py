import contextlib
import copy
import time

try:
    _TIMER = time.monotonic_ns
    _NS_TIMER = True
except AttributeError:
    _TIMER = time.monotonic
    _NS_TIMER = False

class PerfRegions:
    def __init__(self, names = []):
        self.times = dict((name, 0) for name in names)
        self._activeset = set()

        self._starttimes = dict((name, None) for name in names)

    def start_region(self, name):
        assert name in self.times
        assert name not in self._activeset
        self._activeset.add(name)
        self._starttimes[name] = _TIMER()

    def stop_region(self, name):
        stop_t = _TIMER()
        assert name in self._activeset

        elapsed = stop_t - self._starttimes[name]
        self.times[name] += elapsed
        self._activeset.remove(name)

    @contextlib.contextmanager
    def region(self, name):
        try:
            self.start_region(name)
            yield
        finally:
            self.stop_region(name)

    def times_ns(self):
        assert len(self._activeset) == 0
        if _NS_TIMER:
            return copy.copy(self.times)
        else:
            return dict((k,v*1e9) for k,v in self.times.items())

    def times_us(self):
        return dict((k,v/1000) for k,v in self.times_ns().items())

    def times_sec(self):
        return dict((k,v/1e9) for k,v in self.times_ns().items())

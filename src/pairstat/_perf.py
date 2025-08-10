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
    def __init__(self, names=[]):
        self.times = dict((name, 0) for name in names)
        self._activeset = set()

        self._starttimes = dict((name, None) for name in self.times)

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

    def __add__(self, other):
        if not isinstance(other, PerfRegions):
            return NotImplemented
        elif self.any_active_regions() or other.any_active_regions():
            raise RuntimeError(
                "__add__ was called for a PerfRegion which has an active region"
            )
        elif len(self.times) != len(other.times):
            raise ValueError("operands don't have the same region names")

        out_times = {}
        try:
            for k in self.times:
                out_times[k] = self.times[k] + other.times[k]
        except KeyError:
            raise ValueError("operands don't have the same region names")
        out = PerfRegions(out_times.keys())
        out.times = out_times
        return out

    def any_active_regions(self):
        return len(self._activeset) > 0

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
            return dict((k, v * 1e9) for k, v in self.times.items())

    def times_us(self):
        return dict((k, v / 1000) for k, v in self.times_ns().items())

    def times_sec(self):
        return dict((k, v / 1e9) for k, v in self.times_ns().items())

    def summarize_timing_sec(self):
        times = self.times_sec()
        return "    ".join(f"{key}: {val:>15}" for key, val in times.items())

    def __getstate__(self):
        """
        Modified pickling behavior since this object is stateful
        """
        assert not self.any_active_regions()
        state = self.__dict__.copy()

        # remove the unpicklable entries
        del state["_activeset"]
        del state["_starttimes"]
        return state

    def __setstate__(self, state):
        # restore instance attributes
        self.__dict__.update(state)
        # initialize stateful attributes
        self._activeset = set()
        self._starttimes = dict((name, None) for name in self.times)

import yt

# this is pretty hacky!
_CACHED_DATA = None

class SnapDatasetInitializer:
    """
    This is supposed to manage the creation of a yt-dataset (to help
    facillitate pickling).

    Parameters
    ----------
    fname: str subclass instance of 
        File name of a loadable simulation dataset.
    setup_func: callable, optional
        This is a callable that accepts a dataset object as an argument. This
        Should only be specified if the filename was specified as the first 
        argument. This should be picklable (e.g. it can't be a lambda function
        or a function defined in a function).
    """

    def __init__(self, fname, setup_func = None):
        self._fname = fname
        assert (setup_func is None) or callable(setup_func)
        self._setup_func = setup_func

    def __call__(self):
        """
        Initializes a snapshot dataset object.

        Returns
        -------
        out: instance of a subclass of `yt.data_objects.static_output.Dataset`
        """
        # the way we cache data is super hacky! But it's the only way to do it
        # without pickling the cached data.
        global _CACHED_DATA
        cached_instance = (('_CACHED_DATA' in globals()) and
                           (_CACHED_DATA is not None) and
                           (_CACHED_DATA[0] == self._fname))
        if cached_instance:
            ds = _CACHED_DATA[1]
        else:
            if _CACHED_DATA is not None:
                _CACHED_DATA[1].index.clear_all_data() # keep memory usage down!
            ds = yt.load(self._fname)
            func = self._setup_func
            func(ds)
            _CACHED_DATA = (self._fname,ds)
        return ds

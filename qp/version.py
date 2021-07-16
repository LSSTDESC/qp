def find_from_setuptools_scm():
    # setuptools_scm should install a
    # file _version alongside this one.
    # This is the fastest method
    from . import _version
    return _version.__version__

def find_from_metadata(): # pragma: no cover
    # this works on python 3.8 and above
    # and is faster than the version below,
    # so we try it first
    from importlib.metadata import version
    return version = version("qp")

def find_from_pkg_resources(): # pragma: no cover
    # This is slower than the above but works
    # on older versions as long as pkg_resources
    # is available. It comes with setuptools
    # so it should be
    from pkg_resources import get_distribution
    return get_distribution("qp").version

def find_version(): # pragma: no cover
    fs = [find_from_setuptools_scm, find_from_metadata, find_from_pkg_resources]

    for f in fs:
        try:
            v = f()
            break
        except:
            pass
    # Fall back option!
    else:
        v = "unknown"
    return v

__version__ = find_version()

def find_version():
    # setuptools_scm should install a
    # file _version alongside this one.
    # This is the fastest method
    from . import _version
    return _version.version


try:
    __version__ = find_version()
except:
    __version__ = "unknown"

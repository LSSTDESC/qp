def find_version():
    # setuptools_scm should install a
    # file _version alongside this one.
    from . import _version  # pylint: disable=import-outside-toplevel

    return _version.version


try:
    __version__ = find_version()
except:  # pragma: no cover  # pylint: disable=bare-except
    __version__ = "unknown"

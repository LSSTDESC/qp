""" Lazy loading modules """

from tables_io.lazy_modules import lazyImport

mpl = lazyImport("matplotlib")
plt = lazyImport("matplotlib.pyplot")
mixture = lazyImport("sklearn.mixture")

""" Lazy loading modules """

from tables_io.lazy_modules import lazyImport

mpl = lazyImport('matploblib')
plt = lazyImport('matplotlib.pyplot')
sklearn = lazyImport('sklearn')

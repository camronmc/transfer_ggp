from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        'persistent_array.pyx',
        'node.pyx',
        'propnet.pyx'
        ], annotate=True)
)

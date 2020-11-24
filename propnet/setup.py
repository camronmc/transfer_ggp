from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        'cammcts.pyx',
        'persistent_array.pyx',
        'node.pyx',
        'propnet.pyx'
        ], annotate=True)
)

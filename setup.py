from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'auxiliary',
    ext_modules = cythonize("auxiliary.pyx"),
)

setup(
    name = 'GaloisField',
    ext_modules = cythonize("GaloisField.pyx"),
)

setup(
  name = 'ReedSolomon',
  ext_modules = cythonize("ReedSolomon.pyx"),
) 

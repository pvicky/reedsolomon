from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("ReedSolomon", ["ReedSolomon.pyx"]),
    Extension("GaloisField", ["GaloisField.pyx"]),
    Extension("auxiliary", ["auxiliary.pyx"]),
]

setup(
    name = "RSEncoderDecoder",
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
 

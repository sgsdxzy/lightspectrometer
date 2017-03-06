from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "spectrometer",
    version = 0.2,
    
    ext_modules = cythonize('spectrometer.pyx'),
)


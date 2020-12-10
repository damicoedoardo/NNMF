try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext


import numpy
import sys
import re

def compile_cython(file):

    extensionName = re.sub("\.pyx", "", file).replace('/', '.')

    ext_modules = Extension(extensionName,
                [file],
                include_dirs=[numpy.get_include(),],
                )

    print(extensionName)
    print(file)

    setup(name='prova',
          ext_modules=cythonize(ext_modules, annotate=True))
          

if __name__ == "__main__":

    # Get the name of the file to compile
    fileToCompile = sys.argv[1]
    # Remove the argument from sys argv in order for it to contain only what setup needs
    del sys.argv[1]

    compile_cython(fileToCompile)
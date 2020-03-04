# setup.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

models_dir = os.path.dirname(__file__) or os.getcwd()
setup(
    ext_modules = cythonize(models_dir+"/word2vec_inner.pyx"),
    include_dirs = [models_dir, numpy.get_include()],
    build_dir="./"
)

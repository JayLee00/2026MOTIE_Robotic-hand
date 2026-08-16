import os
from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

numpy_include_dir = numpy.get_include()

triangle_hash_module = Extension(
    'anyplace.utils.mesh_util.triangle_hash',
    sources=[
        os.path.join('anyplace', 'utils', 'mesh_util', 'triangle_hash.pyx'),
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir],
    language='c++'
)

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_requirements_file(filename):
    req_file_path = os.path.join(dir_path, filename)
    with open(req_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

packages = find_packages()
for p in packages:
    assert p == 'anyplace' or p.startswith('anyplace.')

# Setup
setup(
    name='anyplace',
    author='Yuchi (Allan) Zhao',
    license='MIT',
    packages=packages,
    install_requires=read_requirements_file('requirements.txt'),
    ext_modules=cythonize([triangle_hash_module]),
)


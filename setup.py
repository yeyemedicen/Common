import os
from distutils.core import setup
from codecs import open

from common import __version__

'''
    Install with
        $ pip install --user
    or
        $ pip install -e . --user
    for an editable development build (no need to reinstall package after
    modifications).
'''

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='common',
    packages=['common'],
    version=__version__,
    description='Common functions for FEniCS based solvers',
    long_description=long_description,
    author='David Nolte',
    author_email='dnolte@dim.uchile.cl',
    url='http://gitlab.dim.uchile.cl/biomed/Common',
    # requires=['numpy (>=1.7)', 'scipy (>=0.13)'],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    entry_points={
        'console_scripts': [
            'xml2hdf5 = common.xml2hdf5:_main']
    }
)

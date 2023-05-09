from setuptools import setup
import re
import os
import sys

ver_info = sys.version_info
if ver_info < (3,7,0):
    raise RuntimeError("HapASeg requires at least python 3.7")

setup(
    name = 'HapASeg',
    version = "0.1",
    packages = [
        'hapaseg',
    ],
    scripts = ['benchmarking/generate_sim_data.py',
               'benchmarking/compare_outputs.py',
               'benchmarking/preprocess_raw_data.py'],
#    package_data={
#        '':[
#            'backends/slurm-gcp/*',
#            'backends/slurm-gcp/scripts/*',
#            'backends/slurm-docker/src/*',
#            'backends/dummy/*',
#            'backends/dummy/conf/*',
#            'localization/debug.sh'
#        ],
#    },
    entry_points={
        'console_scripts':[
            'hapaseg = hapaseg.__main__:main',
        ]
    },
    description = '',
    url = '',
    author = 'Julian Hess',
    author_email = 'jhess@broadinstitute.org',
    #long_description = long_description,
    #long_description_content_type = 'text/markdown',
    install_requires = [
        'pandas==1.4.2',
        'numpy>=1.18.0, <1.23.0', # <1.23.0 to satisfy weird scipy incompatability
        'more-itertools>=8.10.0',
        'numpy_groupies>=0.9.14',
        'h5py'
    ],
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: System :: Clustering",
        "Topic :: System :: Distributed Computing",
        "Typing :: Typed",
        "License :: OSI Approved :: BSD License"
    ],
    license="BSD3"
)

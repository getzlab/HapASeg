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
        'hapaseg_local'
    ],

    scripts = ['benchmarking/generate_sim_data.py',
               'benchmarking/compare_outputs.py',
               'benchmarking/preprocess_raw_data.py',
               'hapaseg_local/hg19_download_1kg.sh',
               'hapaseg_local/hg38_download_1kg.sh'],

    entry_points={
        'console_scripts':[
            'hapaseg = hapaseg.__main__:main',
            'hapaseg_local = hapaseg_local.hapaseg_local:run_hapaseg_local',
            'hapaseg_local_install_ref_files = hapaseg_local.install_ref_files:download_ref_files'
        ]
    },
    description = '',
    url = '',
    author = 'Oliver Priebe, Julian Hess',
    author_email = 'opriebe@broadinstitute.org, jhess@broadinstitute.org',

    install_requires = [
        'pandas==1.4.2',
        'numpy>=1.18.0, <1.23.0', # <1.23.0 to satisfy weird scipy incompatability
        'more-itertools>=8.10.0',
        'numpy_groupies>=0.9.14',
        'h5py',
        'pyyaml'
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

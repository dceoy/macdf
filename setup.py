#!/usr/bin/env python

from setuptools import find_packages, setup

from macdf import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='macdf',
    version=__version__,
    author='Daichi Narushima',
    author_email='dnarsil+github@gmail.com',
    description='MACD-based Forex Trader',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dceoy/macdf',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'docopt', 'numpy', 'oanda-cli', 'pandas', 'pyyaml', 'redis',
        'scikit-learn', 'statsmodels', 'v20'
    ],
    entry_points={'console_scripts': ['macdf=macdf.cli:main']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Office/Business :: Financial :: Investment'
    ],
    python_requires='>=3.6'
)

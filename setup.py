#!/usr/bin/env python

import setuptools

from io import open

requirements = ('scipy>=1.0.0', 'numpy')

classifiers = ('Development Status :: 4 - Beta',
			   'Intended Audience :: Science/Research',
			   'License :: OSI Approved :: BSD License',
			   'Programming Language :: Python',
			   'Programming Language :: Python :: 2.7',
			   'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
			   'Programming Language :: Python :: 3.6',
			   'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
			   'Topic :: Scientific/Engineering :: Physics',
			   'Topic :: Software Development :: Libraries',
			   'Operating System :: OS Independent')

with open("README.md", 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='fronts',
    version='0.9.7',
    author="Gabriel S. Gerlero",
    description="Numerical library for one-dimensional nonlinear diffusion problems in semi-infinite domains",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/gerlero/fronts",
    project_urls={
        "Documentation": "https://fronts.readthedocs.io",
        "Bug Tracker": "https://github.com/gerlero/fronts/issues",
        "Source Code": "https://github.com/gerlero/fronts",
    },
    install_requires=requirements,
    packages=('fronts',),
    license='BSD',
    classifiers=classifiers,
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*'
)
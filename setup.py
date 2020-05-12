#!/usr/bin/env python

import setuptools

from io import open
import re

with open("README.md", 'r', encoding='utf-8') as f:
    readme = f.read()

with open("fronts/__init__.py", 'r', encoding='utf-8') as f:
    init = f.read()

version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init, re.M).group(1)
# Parse instead of import because we may not have dependencies available yet.

extras = {
    'examples': ['matplotlib'],
    'symbolic': ['sympy'],
    'doc': ['sphinx']
}
extras['dev'] = extras['examples'] + extras['symbolic'] + extras['doc']

setuptools.setup(
    name='fronts',
    version=version,
    author="Gabriel S. Gerlero",
    author_email="ggerlero@cimec.unl.edu.ar",
    description="Numerical library for nonlinear diffusion problems based on " 
                "the Boltzmann transformation.",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/gerlero/fronts",
    project_urls={
        "Documentation": "https://fronts.readthedocs.io",
        "Bug Tracker": "https://github.com/gerlero/fronts/issues",
        "Source Code": "https://github.com/gerlero/fronts",
    },
    packages=['fronts'],
    license='BSD',
    classifiers=['Development Status :: 4 - Beta',
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
                 'Operating System :: OS Independent'],
    install_requires=['scipy>=1.0.0', 'numpy'],
    extras_require=extras,
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    options={"bdist_wheel": {"universal": "1"}}
)

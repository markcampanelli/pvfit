"""
This is solely to support side-by-side wheel builds with nuitka using legacy setuptools setup.

This is a workaround because two different build backends (i.e., setuptools vs. nuitka) cannot apparently be selected
dynamically using pyproject.toml.
"""

from setuptools import setup

# setuptools_scm version should match pyproject.toml, which is not used in this build path.
setup(setup_requires=["setuptools_scm[toml]~=6.4"], use_scm_version=True)

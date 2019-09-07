from pkg_resources import get_distribution

# This relies on setuptools_scm magic in setup.py, and raises when package is not installed.
__version__ = get_distribution(__name__).version

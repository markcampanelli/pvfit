import logging

# Set up global logging before doing anything else.
logging.basicConfig(level=logging.INFO)

from pvfit.common.utils import get_version

__version__ = get_version()

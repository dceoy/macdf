#!/usr/bin/env python
"""
MACD-based Forex Trader using oanda-cli

Usage:
    macdf -h|--help
    macdf --version
    macdf [--debug|--info] <arg>...

Options:
    -h, --help          Print help and exit
    --version           Print version and exit
    --debug, --info     Execute a command with debug|info messages
"""

import logging
import os

from docopt import docopt

from . import __version__


def main():
    args = docopt(__doc__, version=__version__)
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    print(args)


def _set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )

#!/usr/bin/env python
"""
MACD-based Forex Trader using Oanda API

Usage:
    macdf -h|--help
    macdf --version
    macdf close [--debug|--info] [--account=<id>] [--token=<str>] [--env=<str>]
        [<instrument>...]
    macdf trade [--debug|--info] [--account=<id>] [--token=<str>] [--env=<str>]
        [--quiet] [--dry-run] <instrument>...

Options:
    -h, --help          Print help and exit
    --version           Print version and exit
    --debug, --info     Execute a command with debug|info messages
    --account=<id>      Specify an account ID [$OANDA_ID]
    --token=<str>       Specify an API token [$OANDA_TOKEN]
    --env=<str>         Specify a trading environment [default: trade]
    --quiet             Suppress messages
    --dry-run           Invoke a trader with dry-run mode

Commands:
    close               Close positions (if not <instrument>, close all)
    trade               Trade currencies

Arguments:
    <instrument>        Forex instrument, such as EUR_USD and USD_JPY
"""

import logging
import os

from docopt import docopt

from . import __version__
from .order import close_positions
from .util import set_log_config


def main():
    args = docopt(__doc__, version=f'macdf {__version__}')
    set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    account_id = (args['--account'] or os.getenv('OANDA_ID'))
    api_token = (args['--token'] or os.getenv('OANDA_TOKEN'))
    if args.get('close'):
        close_positions(
            account_id=account_id, api_token=api_token, environment='trade',
            instruments=args['<instrument>']
        )
    elif args.get('trade'):
        pass

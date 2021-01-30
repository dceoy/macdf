#!/usr/bin/env python
"""
MACD-based Forex Trader

Usage:
    macdf -h|--help
    macdf --version
    macdf init [--debug|--info] [--file=<yaml>]
    macdf close [--debug|--info] [--file=<yaml>] [<instrument>...]
    macdf trade [--debug|--info] [--file=<yaml>] [--quiet] [--dry-run]
                [<instrument>...]

Options:
    -h, --help          Print help and exit
    --version           Print version and exit
    --debug, --info     Execute a command with debug|info messages
    --file=<yaml>       Set a path to a YAML for configurations [$OANDA_YML]
    --quiet             Suppress messages
    --dry-run           Invoke a trader with dry-run mode

Commands:
    init                Create a YAML template for configuration
    close               Close positions (if not <instrument>, close all)
    open                Invoke an autonomous trader

Arguments:
    <info_target>       { instruments, prices, account, accounts, orders,
                          trades, positions, position, order_book,
                          position_book }
    <instrument>        { AUD_CAD, AUD_CHF, AUD_HKD, AUD_JPY, AUD_NZD, AUD_SGD,
                          AUD_USD, CAD_CHF, CAD_HKD, CAD_JPY, CAD_SGD, CHF_HKD,
                          CHF_JPY, CHF_ZAR, EUR_AUD, EUR_CAD, EUR_CHF, EUR_CZK,
                          EUR_DKK, EUR_GBP, EUR_HKD, EUR_HUF, EUR_JPY, EUR_NOK,
                          EUR_NZD, EUR_PLN, EUR_SEK, EUR_SGD, EUR_TRY, EUR_USD,
                          EUR_ZAR, GBP_AUD, GBP_CAD, GBP_CHF, GBP_HKD, GBP_JPY,
                          GBP_NZD, GBP_PLN, GBP_SGD, GBP_USD, GBP_ZAR, HKD_JPY,
                          NZD_CAD, NZD_CHF, NZD_HKD, NZD_JPY, NZD_SGD, NZD_USD,
                          SGD_CHF, SGD_HKD, SGD_JPY, TRY_JPY, USD_CAD, USD_CHF,
                          USD_CNH, USD_CZK, USD_DKK, USD_HKD, USD_HUF, USD_INR,
                          USD_JPY, USD_MXN, USD_NOK, USD_PLN, USD_SAR, USD_SEK,
                          USD_SGD, USD_THB, USD_TRY, USD_ZAR, ZAR_JPY }
    <data_path>         Path to an input CSV or SQLite file
    <graph_path>        Path to an output graphics file such as PDF or PNG
"""

import logging
import os
from pathlib import Path

from docopt import docopt
from oandacli.cli.main import execute_command
from oandacli.util.config import fetch_config_yml_path, write_config_yml
from oandacli.util.logger import set_log_config

from .. import __version__
from ..call.trader import invoke_trader


def main():
    args = docopt(__doc__, version=f'macdf {__version__}')
    set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    config_yml_path = fetch_config_yml_path(
        path=args['--file'], env='MACDF_YML', default='macdf.yml'
    )
    if args['init']:
        write_config_yml(
            dest_path=config_yml_path,
            template_path=str(
                Path(__file__).parent.parent.joinpath(
                    'static/default_macdf.yml'
                ).resolve()
            )
        )
    elif args['open']:
        invoke_trader(
            config_yml=config_yml_path, instruments=args['<instrument>'],
            model=args['--model'], interval_sec=args['--interval'],
            timeout_sec=args['--timeout'], standalone=args['--standalone'],
            redis_host=args['--redis-host'], redis_port=args['--redis-port'],
            redis_db=args['--redis-db'], log_dir_path=args['--log-dir'],
            ignore_api_error=args['--ignore-api-error'], quiet=args['--quiet'],
            dry_run=args['--dry-run']
        )
    else:
        execute_command(args=args, config_yml_path=config_yml_path)

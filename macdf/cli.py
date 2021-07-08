#!/usr/bin/env python
"""
MACD-based Forex Trader using Oanda API

Usage:
    macdf -h|--help
    macdf --version
    macdf close [--debug|--info] [--oanda-account=<id>] [--oanda-token=<str>]
        [--oanda-env=<str>] [<instrument>...]
    macdf spread [--debug|--info] [--oanda-account=<id>] [--oanda-token=<str>]
        [--oanda-env=<str>] [--csv=<path>] [--quiet] [<instrument>...]
    macdf trade [--debug|--info] [--oanda-account=<id>] [--oanda-token=<str>]
        [--oanda-env=<str>] [--quiet] [--dry-run] [--retry-count=<int>]
        [--granularity=<str>] [--betting-strategy=<str>]
        [--scanned-transaction-count=<int>] [--sleeping=<ratio>]
        [--unit-margin=<ratio>] [--preserved-margin=<ratio>]
        [--take-profit-limit=<float>] [--trailing-stop-limit=<float>]
        [--stop-loss-limit=<float>] [--max-spread=<float>]
        [--fast-ema-span=<int>] [--slow-ema-span=<int>] [--macd-ema-span=<int>]
        [--generic-ema-span=<int>] [--significance-level=<float>]
        [--trigger-sharpe-ratio=<float>] [--granularity-scorer=<str>]
        <instrument>...

Options:
    -h, --help              Print help and exit
    --version               Print version and exit
    --debug, --info         Execute a command with debug|info messages
    --oanda-account=<id>    Set an Oanda account ID [$OANDA_ACCOUNT]
    --oanda-token=<str>     Set an Oanda API token [$OANDA_TOKEN]
    --oanda-env=<str>       Set an Oanda trading environment [default: trade]
                            { trade, practice }
    --csv=<path>            Write data with CSV into a file
    --quiet                 Suppress messages
    --dry-run               Invoke a trader with dry-run mode
    --retry-count=<int>     Set the retry count due to API errors [default: 0]
    --granularity=<str>     Set the granularities [default: D]
                            { S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30,
                              H1, H2, H3, H4, H6, H8, H12, D, W, M }
    --betting-strategy=<str>
                            Set the betting strategy [default: constant]
                            { constant, martingale, paroli, dalembert,
                              oscarsgrind }
    --scanned-transaction-count=<int>
                            Set the transaction count to scan [default: 0]
    --sleeping=<ratio>      Set the daily sleeping ratio [default: 0]
    --unit-margin=<ratio>   Set the unit margin ratio to NAV [default: 0.01]
    --preserved-margin=<ratio>
                            Set the preserved margin ratio [default: 0.01]
    --take-profit-limit=<float>
                            Set the take-profit limit ratio [default: 0.01]
    --trailing-stop-limit=<float>
                            Set the trailing-stop limit ratio [default: 0.01]
    --stop-loss-limit=<float>
                            Set the stop-loss limit ratio [default: 0.01]
    --max-spread=<float>    Set the max spread ratio [default: 0.01]
    --fast-ema-span=<int>   Set the fast EMA span [default: 12]
    --slow-ema-span=<int>   Set the slow EMA span [default: 26]
    --macd-ema-span=<int>   Set the MACD EMA span [default: 9]
    --generic-ema-span=<int>
                            Set the generic EMA span [default: 9]
    --significance-level=<float>
                            Set the significance level [default: 0.01]
    --trigger-sharpe-ratio=<float>
                            Set the trigger Sharpe ratio [default: 1]
    --granularity-scorer=<str>
                            Set the granularity scorer [default: ljungboxtest]
                            { ljungboxtest, sharperatio }

Commands:
    close                   Close positions (if not <instrument>, close all)
    spread                  Print the ratios of spread to price
    trade                   Trade currencies

Arguments:
    <instrument>            Forex instrumnt such as EUR_USD and USD_JPY
"""

import logging
import os

import v20
from docopt import docopt
from oandacli.call.info import print_spread_ratios
from oandacli.call.order import close_positions
from oandacli.util.logger import set_log_config

from . import __version__
from .trader import AutoTrader


def main():
    args = docopt(__doc__, version=f'macdf {__version__}')
    set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    oanda_account_id = (args['--oanda-account'] or os.getenv('OANDA_ACCOUNT'))
    oanda_api_token = (args['--oanda-token'] or os.getenv('OANDA_TOKEN'))
    if args.get('trade'):
        logger.info('Autonomous trading')
        AutoTrader(
            oanda_account_id=oanda_account_id, oanda_api_token=oanda_api_token,
            oanda_environment=args['--oanda-env'],
            retry_count=int(args['--retry-count']),
            instruments=args['<instrument>'],
            granularities=args['--granularity'].split(','),
            betting_strategy=args['--betting-strategy'],
            scanned_transaction_count=int(args['--scanned-transaction-count']),
            sleeping_ratio=float(args['--sleeping']),
            unit_margin_ratio=float(args['--unit-margin']),
            preserved_margin_ratio=float(args['--preserved-margin']),
            take_profit_limit_ratio=float(args['--take-profit-limit']),
            trailing_stop_limit_ratio=float(args['--trailing-stop-limit']),
            stop_loss_limit_ratio=float(args['--stop-loss-limit']),
            max_spread_ratio=float(args['--max-spread']),
            fast_ema_span=int(args['--fast-ema-span']),
            slow_ema_span=int(args['--slow-ema-span']),
            macd_ema_span=int(args['--macd-ema-span']),
            generic_ema_span=int(args['--generic-ema-span']),
            significance_level=float(args['--significance-level']),
            trigger_sharpe_ratio=float(args['--trigger-sharpe-ratio']),
            granularity_scorer=args['--granularity-scorer'], log_dir_path=None,
            quiet=args['--quiet'], dry_run=args['--dry-run']
        ).invoke()
    else:
        oanda_api = _create_oanda_api(
            api_token=oanda_api_token, environment=args['--oanda-env']
        )
        if args.get('close'):
            close_positions(
                api=oanda_api, account_id=oanda_account_id,
                instruments=args['<instrument>']
            )
        elif args.get('spread'):
            print_spread_ratios(
                api=oanda_api, account_id=oanda_account_id,
                instruments=args['<instrument>'], csv_path=args['--csv'],
                quiet=args['--quiet']
            )


def _create_oanda_api(api_token, environment='trade', stream=False, **kwargs):
    return v20.Context(
        hostname='{0}-fx{1}.oanda.com'.format(
            ('stream' if stream else 'api'), environment
        ),
        token=api_token, **kwargs
    )

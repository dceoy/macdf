#!/usr/bin/env python

import json
import logging
import os
import signal
from math import ceil
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from oandacli.util.logger import log_response
from v20 import Context, V20ConnectionError, V20Timeout

from .bet import BettingSystem
from .signal import MacdSignalDetector


class APIResponseError(RuntimeError):
    pass


class OandaTraderCore(object):
    def __init__(self, instruments, oanda_account_id, oanda_api_token,
                 oanda_environment='trade', betting_strategy='constant',
                 scanned_transaction_count=0, unit_margin_ratio=0.01,
                 preserved_margin_ratio=0.01, take_profit_limit_ratio=0.01,
                 trailing_stop_limit_ratio=0.01, stop_loss_limit_ratio=0.01,
                 log_dir_path=None, quiet=False, dry_run=False):
        self.__logger = logging.getLogger(__name__)
        self.__api = Context(
            hostname=f'api-fx{oanda_environment}.oanda.com',
            token=oanda_api_token
        )
        self.__account_id = oanda_account_id
        self.instruments = instruments
        self.betting_system = BettingSystem(strategy=betting_strategy)
        self.__scanned_transaction_count = int(scanned_transaction_count)
        self.__unit_margin_ratio = float(unit_margin_ratio)
        self.__preserved_margin_ratio = float(preserved_margin_ratio)
        self.__take_profit_limit_ratio = float(take_profit_limit_ratio)
        self.__trailing_stop_limit_ratio = float(trailing_stop_limit_ratio)
        self.__stop_loss_limit_ratio = float(stop_loss_limit_ratio)
        self.__quiet = quiet
        self.__dry_run = dry_run
        if log_dir_path:
            log_dir = Path(log_dir_path).resolve()
            self.__log_dir_path = str(log_dir)
            os.makedirs(self.__log_dir_path, exist_ok=True)
            self.__order_log_path = str(log_dir.joinpath('order.json.txt'))
            self.__txn_log_path = str(log_dir.joinpath('txn.json.txt'))
        else:
            self.__log_dir_path = None
            self.__order_log_path = None
            self.__txn_log_path = None
        self.__anchored_txn_id = None
        self.__last_txn_id = None
        self.pos_dict = dict()
        self.balance = None
        self.margin_avail = None
        self.__account_currency = None
        self.txn_list = list()
        self.__inst_dict = dict()
        self.price_dict = dict()
        self.unit_costs = dict()

    def _refresh_account_dicts(self):
        res = self.__api.account.get(accountID=self.__account_id)
        if 'account' in res.body and 'lastTransactionID' in res.body:
            acc = res.body['account']
            if not self.__anchored_txn_id:
                self.__anchored_txn_id = max(
                    0,
                    int(res.body['lastTransactionID'])
                    - self.__scanned_transaction_count
                )
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )
        self.balance = float(acc.balance)
        self.margin_avail = float(acc.marginAvailable)
        self.__account_currency = acc.currency
        self.pos_dict = {
            p.instrument: (
                {'side': 'long', 'units': int(p.long.units)} if p.long.tradeIDs
                else {'side': 'short', 'units': int(p.short.units)}
            ) for p in acc.positions if p.long.tradeIDs or p.short.tradeIDs
        }

    def _place_order(self, closing=False, **kwargs):
        if closing:
            p = self.pos_dict.get(kwargs['instrument'])
            f_args = {
                'accountID': self.__account_id, **kwargs,
                **{
                    f'{k}Units': ('ALL' if p and p['side'] == k else 'NONE')
                    for k in ['long', 'short']
                }
            }
        else:
            f_args = {'accountID': self.__account_id, **kwargs}
        if self.__dry_run:
            self.__logger.info(
                os.linesep + pformat({
                    'func': ('position.close' if closing else 'order.create'),
                    'args': f_args
                })
            )
        else:
            if closing:
                res = self.__api.position.close(**f_args)
            else:
                res = self.__api.order.create(**f_args)
            log_response(res, logger=self.__logger)
            if not (100 <= res.status <= 399):
                raise APIResponseError(
                    'unexpected response:' + os.linesep + pformat(res.body)
                )
            elif self.__order_log_path:
                self._write_data(res.raw_body, path=self.__order_log_path)

    def refresh_oanda_dicts(self):
        self._refresh_account_dicts()
        self._refresh_txn_list()
        self._refresh_inst_dict()
        self._refresh_price_dict()
        self._refresh_unit_costs()

    def _refresh_txn_list(self):
        init = (not self.__last_txn_id)
        res = self.__api.transaction.since(
            accountID=self.__account_id,
            id=(self.__anchored_txn_id if init else self.__last_txn_id)
        )
        if 'lastTransactionID' in res.body:
            self.__last_txn_id = res.body['lastTransactionID']
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )
        if res.body.get('transactions'):
            t_new = [t.dict() for t in res.body['transactions']]
            self.__logger.info('transactions:' + os.linesep + pformat(t_new))
            self.txn_list = self.txn_list + t_new
            if not init and t_new:
                if self.__txn_log_path:
                    self._write_data(
                        json.dumps(t_new), path=self.__txn_log_path
                    )

    def _refresh_inst_dict(self):
        res = self.__api.account.instruments(accountID=self.__account_id)
        if 'instruments' in res.body:
            self.__inst_dict = {
                c.name: vars(c) for c in res.body['instruments']
            }
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )

    def _refresh_price_dict(self):
        res = self.__api.pricing.get(
            accountID=self.__account_id,
            instruments=','.join(self.__inst_dict.keys())
        )
        if 'prices' in res.body:
            self.price_dict = {
                p.instrument: {
                    'bid': p.closeoutBid, 'ask': p.closeoutAsk,
                    'tradeable': p.tradeable
                } for p in res.body['prices']
            }
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )

    def _refresh_unit_costs(self):
        self.unit_costs = {
            i: self._calculate_bp_value(instrument=i) * float(e['marginRate'])
            for i, e in self.__inst_dict.items() if i in self.instruments
        }

    def _calculate_bp_value(self, instrument):
        cur_pair = instrument.split('_')
        if cur_pair[0] == self.__account_currency:
            bpv = 1 / self.price_dict[instrument]['ask']
        elif cur_pair[1] == self.__account_currency:
            bpv = self.price_dict[instrument]['ask']
        else:
            bpv = None
            for i in self.__inst_dict.keys():
                if bpv:
                    break
                elif i == cur_pair[0] + '_' + self.__account_currency:
                    bpv = self.price_dict[i]['ask']
                elif i == self.__account_currency + '_' + cur_pair[0]:
                    bpv = 1 / self.price_dict[i]['ask']
                elif i == cur_pair[1] + '_' + self.__account_currency:
                    bpv = (
                        self.price_dict[instrument]['ask']
                        * self.price_dict[i]['ask']
                    )
                elif i == self.__account_currency + '_' + cur_pair[1]:
                    bpv = (
                        self.price_dict[instrument]['ask']
                        / self.price_dict[i]['ask']
                    )
            assert bpv, f'bp value calculatiton failed: {instrument}'
        return bpv

    def design_and_place_order(self, instrument, act):
        pos = self.pos_dict.get(instrument)
        if pos and act and (act == 'closing' or act != pos['side']):
            self.__logger.info('Close a position: {}'.format(pos['side']))
            self._place_order(closing=True, instrument=instrument)
            self._refresh_txn_list()
        if act in ['long', 'short']:
            limits = self._design_order_limits(instrument=instrument, side=act)
            self.__logger.debug(f'limits: {limits}')
            units = self._design_order_units(instrument=instrument, side=act)
            self.__logger.debug(f'units: {units}')
            if units != 0:
                self.__logger.info(f'Open an order: {act}')
                self._place_order(
                    order={
                        'type': 'MARKET', 'instrument': instrument,
                        'units': str(units), 'timeInForce': 'FOK',
                        'positionFill': 'DEFAULT', **limits
                    }
                )
                self._refresh_txn_list()
            else:
                self.__logger.info(f'Skip an order: {act}')
            return units
        else:
            return 0

    def _design_order_limits(self, instrument, side):
        ie = self.__inst_dict[instrument]
        r = self.price_dict[instrument][{'long': 'ask', 'short': 'bid'}[side]]
        ts_range = [
            float(ie['minimumTrailingStopDistance']),
            float(ie['maximumTrailingStopDistance'])
        ]
        ts_dist_ratio = int(r * self.__trailing_stop_limit_ratio / ts_range[0])
        if ts_dist_ratio <= 1:
            trailing_stop = ie['minimumTrailingStopDistance']
        else:
            ts_dist = np.float16(ts_range[0] * ts_dist_ratio)
            if ts_dist >= ts_range[1]:
                trailing_stop = ie['maximumTrailingStopDistance']
            else:
                trailing_stop = str(ts_dist)
        tif = {'timeInForce': 'GTC'}
        return {
            'takeProfitOnFill': {
                'price': str(
                    np.float16(
                        r + r * self.__take_profit_limit_ratio
                        * {'long': 1, 'short': -1}[side]
                    )
                ),
                **tif
            },
            'stopLossOnFill': {
                'price': str(
                    np.float16(
                        r + r * self.__stop_loss_limit_ratio
                        * {'long': -1, 'short': 1}[side]
                    )
                ),
                **tif
            },
            'trailingStopLossOnFill': {'distance': trailing_stop, **tif}
        }

    def _design_order_units(self, instrument, side):
        max_size = int(self.__inst_dict[instrument]['maximumOrderUnits'])
        avail_size = max(
            ceil(
                (
                    self.margin_avail - self.balance *
                    self.__preserved_margin_ratio
                ) / self.unit_costs[instrument]
            ), 0
        )
        self.__logger.debug(f'avail_size: {avail_size}')
        unit_size = ceil(
            self.balance * self.__unit_margin_ratio
            / self.unit_costs[instrument]
        )
        self.__logger.debug(f'unit_size: {unit_size}')
        bet_size = self.betting_system.calculate_size_by_pl(
            unit_size=unit_size,
            inst_pl_txns=[
                t for t in self.txn_list if (
                    t.get('instrument') == instrument and t.get('pl') and
                    t.get('units')
                )
            ]
        )
        self.__logger.debug(f'bet_size: {bet_size}')
        return (
            int(min(bet_size, avail_size, max_size)) *
            {'long': 1, 'short': -1}[side]
        )

    def print_log(self, data):
        if self.__quiet:
            self.__logger.info(data)
        else:
            print(data, flush=True)

    def print_state_line(self, df_rate, add_str):
        self.print_log(
            '|{0:^11}|{1:^29}|'.format(
                df_rate['instrument'].iloc[-1],
                'B/A:{:>21}'.format(
                    np.array2string(
                        df_rate[['bid', 'ask']].iloc[-1].values,
                        formatter={'float_kind': lambda f: f'{f:8g}'}
                    )
                )
            ) + (add_str or '')
        )

    def _write_data(self, data, path, mode='a', append_linesep=True):
        with open(path, mode) as f:
            f.write(str(data) + (os.linesep if append_linesep else ''))

    def write_turn_log(self, df_rate, **kwargs):
        i = df_rate['instrument'].iloc[-1]
        df_r = df_rate.drop(columns=['instrument'])
        self._write_log_df(name=f'rate.{i}', df=df_r)
        if kwargs:
            self._write_log_df(
                name=f'sig.{i}', df=df_r.tail(n=1).assign(**kwargs)
            )

    def _write_log_df(self, name, df):
        if self.__log_dir_path and df.size:
            self.__logger.debug(f'{name} df:{os.linesep}{df}')
            p = str(Path(self.__log_dir_path).joinpath(f'{name}.tsv'))
            self.__logger.info(f'Write TSV log: {p}')
            self._write_df(df=df, path=p)

    def _write_df(self, df, path, mode='a'):
        df.to_csv(
            path, mode=mode, sep=(',' if path.endswith('.csv') else '\t'),
            header=(not Path(path).is_file())
        )

    def fetch_candle_df(self, instrument, granularity='S5', count=5000,
                        complete=False):
        res = self.__api.instrument.candles(
            instrument=instrument, price='BA', granularity=granularity,
            count=int(count)
        )
        if 'candles' in res.body:
            return pd.DataFrame([
                {
                    'time': c.time, 'bid': c.bid.c, 'ask': c.ask.c,
                    'volume': c.volume, 'complete': c.complete
                } for c in res.body['candles'] if c.complete or not complete
            ]).assign(
                time=lambda d: pd.to_datetime(d['time']), instrument=instrument
            ).set_index('time', drop=True)
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )

    def fetch_latest_price_df(self, instrument):
        res = self.__api.pricing.get(
            accountID=self.__account_id, instruments=instrument
        )
        if 'prices' in res.body:
            return pd.DataFrame([
                {'time': r.time, 'bid': r.closeoutBid, 'ask': r.closeoutAsk}
                for r in res.body['prices']
            ]).assign(
                time=lambda d: pd.to_datetime(d['time']), instrument=instrument
            ).set_index('time')
        else:
            raise APIResponseError(
                'unexpected response:' + os.linesep + pformat(res.body)
            )


class AutoTrader(OandaTraderCore):
    def __init__(self, granularities='D', granularity_scorer='Ljung-Box test',
                 preserved_margin_ratio=0.01, max_spread_ratio=0.01,
                 ignore_api_error=False, retry=1, fast_ema_span=12,
                 slow_ema_span=26, macd_ema_span=9, **kwargs):
        super().__init__(**kwargs)
        self.__logger = logging.getLogger(__name__)
        self.__ignore_api_error = ignore_api_error
        self.__retry = retry
        self.__granularities = (
            {granularities} if isinstance(granularities, str)
            else set(granularities)
        )
        self.__preserved_margin_ratio = preserved_margin_ratio
        self.__max_spread_ratio = float(max_spread_ratio)
        self.__signal_detector = MacdSignalDetector(
            fast_ema_span=int(fast_ema_span),
            slow_ema_span=int(slow_ema_span),
            macd_ema_span=int(macd_ema_span),
            granularity_scorer=granularity_scorer
        )
        self.__cache_length = min(int(slow_ema_span) * 10, 5000)
        self.__logger.debug('vars(self): ' + pformat(vars(self)))

    def invoke(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        for r in range(self.__retry + 1):
            try:
                for i in self.instruments:
                    self.refresh_oanda_dicts()
                    self.make_decision(instrument=i)
            except (V20ConnectionError, V20Timeout, APIResponseError) as e:
                if self.__ignore_api_error or r < self.__retry:
                    self.__logger.error(e)
                else:
                    raise e
            else:
                break

    def make_decision(self, instrument):
        df_r = self.fetch_latest_price_df(instrument=instrument)
        st = self.determine_sig_state(df_rate=df_r)
        new_units = self.design_and_place_order(
            instrument=instrument, act=st['act']
        )
        if new_units != 0:
            st['state'] = st['state'].replace(
                '-> ',
                '-> {:.1f}% '.format(
                    round(
                        abs(
                            new_units * self.unit_costs[instrument] * 100
                            / self.balance
                        ),
                        1
                    )
                )
            )
        elif st['act'] in ['long', 'short']:
            st['state'] = 'LACK OF FUNDS'
        else:
            pass
        self.print_state_line(
            df_rate=df_r,
            add_str=(st['log_str'] + '{:^27}|'.format(st['state']))
        )
        self.write_turn_log(
            df_rate=df_r, **{k: v for k, v in st.items() if k != 'log_str'}
        )

    def determine_sig_state(self, df_rate):
        i = df_rate['instrument'].iloc[-1]
        pos = self.pos_dict.get(i)
        pos_pct = '{:.1f}%'.format(
            round(
                abs(pos['units'] * self.unit_costs[i] * 100 / self.balance), 1
            ) if pos else 0
        )
        sig = self.__signal_detector.detect(
            history_dict={
                g: self.fetch_candle_df(
                    instrument=i, granularity=g, count=self.__cache_length
                )[['ask', 'bid', 'volume']] for g in self.__granularities
            },
            position_side=(pos['side'] if pos else None)
        )
        if not self.price_dict[i]['tradeable']:
            act = None
            state = 'TRADING HALTED'
        elif pos and sig['act'] == 'closing':
            act = 'closing'
            state = '{0} {1} ->'.format(pos_pct, pos['side'].upper())
        elif int(self.balance) == 0:
            act = None
            state = 'NO FUND'
        elif (pos
              and ((sig['act'] and sig['act'] == pos['side'])
                   or not sig['act'])):
            act = None
            state = '{0} {1}'.format(pos_pct, pos['side'].upper())
        elif self._is_margin_lack(instrument=i):
            act = None
            state = 'LACK OF FUNDS'
        elif self._is_over_spread(df_rate=df_rate):
            act = None
            state = 'OVER-SPREAD'
        elif not sig['act']:
            act = None
            state = '-'
        elif pos:
            act = sig['act']
            state = '{0} {1} -> {2}'.format(
                pos_pct, pos['side'].upper(), sig['act'].upper()
            )
        else:
            act = sig['act']
            state = '-> {}'.format(sig['act'].upper())
        return {
            'act': act, 'state': state,
            **{('sig_act' if k == 'act' else k): v for k, v in sig.items()}
        }

    def _is_margin_lack(self, instrument):
        return (
            not self.pos_dict.get(instrument) and
            self.balance * self.__preserved_margin_ratio >= self.margin_avail
        )

    def _is_over_spread(self, df_rate):
        return (
            df_rate.tail(n=1).pipe(
                lambda d: (d['ask'] - d['bid']) / (d['ask'] + d['bid']) * 2
            ).values[0]
            >= self.__max_spread_ratio
        )

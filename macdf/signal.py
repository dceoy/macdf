#!/usr/bin/env python

import logging
import os
import warnings
from pprint import pformat

import numpy as np
import pandas as pd
import statsmodels.api as sm


class MacdSignalDetector(object):
    def __init__(self, fast_ema_span=12, slow_ema_span=26, macd_ema_span=9,
                 granularity_scorer='Ljung-Box test'):
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        granularity_scorers = ['Ljung-Box test', 'Sharpe ratio']
        matched_scorer = [
            s for s in granularity_scorers if (
                granularity_scorer.lower().replace('-', '').replace(' ', '')
                == s.lower().replace('-', '').replace(' ', '')
            )
        ]
        if matched_scorer:
            self.granularity_scorer = matched_scorer[0]
        else:
            raise ValueError(f'invalid scorer: {granularity_scorer}')
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def detect(self, history_dict, position_side=None):
        feature_dict = {
            g: self._calculate_adjusted_macd(
                mid=d[['ask', 'bid']].dropna().mean(axis=1)
            ).tail(self.macd_ema_span * 2)
            for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        sig = feature_dict[granularity].assign(
            macd_div=lambda d: (d['macd'] - d['macd_ema']),
            adjusted_return=lambda d: (
                (np.exp(np.log(d['mid']).diff()) - 1)
                * (-1 if d['macd'].iloc[-1] < d['macd_ema'].iloc[-1] else 1)
                / d['delta_sec'] * d['delta_sec'].mean()
            )
        ).assign(
            macd_div_diff_ema=lambda d: d['macd_div'].diff().ewm(
                span=self.macd_ema_span, adjust=False
            ).mean(),
            ewm_sharpe_ratio=lambda d: d['adjusted_return'].pipe(
                lambda s: (
                    s.ewm(span=self.macd_ema_span, adjust=False).mean()
                    / s.ewm(span=self.macd_ema_span, adjust=False).std(ddof=1)
                )
            )
        ).iloc[-1].to_dict()
        if sig['macd'] > sig['macd_ema']:
            if sig['macd_div_diff_ema'] > 0 and sig['ewm_sharpe_ratio'] > 1:
                act = 'long'
            elif ((position_side == 'long' and sig['macd_div_diff_ema'] < 0)
                  or position_side == 'short'):
                act = 'closing'
            else:
                act = None
        elif sig['macd'] < sig['macd_ema']:
            if sig['macd_div_diff_ema'] < 0 and sig['ewm_sharpe_ratio'] > 1:
                act = 'short'
            elif ((position_side == 'short' and sig['macd_div_diff_ema'] > 0)
                  or position_side == 'long'):
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig,
            'log_str': '{:^51}|'.format(
                '{0} EMSR [MACD/EMA]:{1:>9}{2:>18}'.format(
                    self._parse_granularity(granularity=granularity),
                    '{:.1g}'.format(sig['ewm_sharpe_ratio']),
                    np.array2string(
                        np.array([sig['macd'], sig['macd_ema']]),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    def _calculate_adjusted_macd(self, mid):
        return mid.to_frame(name='mid').reset_index().assign(
            mid_ff=lambda d: d['mid'].fillna(method='ffill'),
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).set_index('time').assign(
            macd=lambda d: (
                d['mid_ff'].ewm(span=self.fast_ema_span, adjust=False).mean()
                - d['mid_ff'].ewm(span=self.slow_ema_span, adjust=False).mean()
            ) / d['delta_sec'] * d['delta_sec'].mean()
        ).drop(columns='mid_ff').assign(
            macd_ema=lambda d:
            d['macd'].ewm(span=self.macd_ema_span, adjust=False).mean()
        )

    def _calculate_hv(self, df):
        return np.log(
            df[['ask', 'bid']].mean(axis=1, skipna=True)
        ).diff().rolling(
            window=self.macd_ema_span
        ).std(ddof=1, skipna=True)

    def _select_best_granularity(self, feature_dict):
        if len(feature_dict) == 1:
            granularity = list(feature_dict.keys())[0]
        elif self.granularity_scorer == 'Ljung-Box test':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                df_g = pd.DataFrame([
                    {
                        'granularity': g,
                        'pvalue': sm.stats.diagnostic.acorr_ljungbox(
                            x=(d['macd'] - d['macd_ema'])
                        )[1][0]
                    } for g, d in feature_dict.items()
                ])
            best_g = df_g.pipe(lambda d: d.iloc[d['pvalue'].idxmin()])
            self.__logger.debug('pvalue: {}'.format(best_g['pvalue']))
            granularity = best_g['granularity']
        elif self.granularity_scorer == 'Sharpe ratio':
            df_g = pd.DataFrame([
                {
                    'granularity': g,
                    'sharpe_ratio': self._calculate_sharpe_ratio(
                        df=d,
                        is_short=(d['macd'].iloc[-1] < d['macd_ema'].iloc[-1])
                    )
                } for g, d in feature_dict.items()
            ])
            best_g = df_g.pipe(lambda d: d.iloc[d['sharpe_ratio'].idxmax()])
            self.__logger.debug(
                'sharpe_ratio: {}'.format(best_g['sharpe_ratio'])
            )
            granularity = best_g['granularity']
        else:
            raise ValueError(f'invalid scorer: {self.granularity_scorer}')
        self.__logger.debug(f'granularity: {granularity}')
        return granularity

    @staticmethod
    def _calculate_sharpe_ratio(df, is_short=False):
        return df.reset_index().assign(
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).assign(
            adjusted_return=lambda d: (
                (np.exp(np.log(d['mid']).diff()) - 1) * (-1 if is_short else 1)
                / d['delta_sec'] * d['delta_sec'].mean()
            )
        )['adjusted_return'].dropna().pipe(
            lambda s: (s.mean() / s.std(ddof=1))
        )

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

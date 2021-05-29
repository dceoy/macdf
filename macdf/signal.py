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
            macd_div=lambda d: d['macd'] - d['macd_ema']
        ).assign(
            macd_div_diff=lambda d: d['macd_div'].diff()
        ).assign(
            macd_div_diff_ema=lambda d: d['macd_div_diff'].ewm(
                span=self.macd_ema_span, adjust=False
            ).mean()
        ).iloc[-1].to_dict()
        df_rate = history_dict[granularity]
        if sig['macd_div'] > 0:
            if ((sig['macd_div_diff_ema'] > 0)
                    or (self._has_high_volume(df=df_rate)
                        and self._has_high_volatility(df=df_rate))):
                act = 'long'
            elif ((position_side == 'long' and sig['macd_div_diff_ema'] < 0)
                  or position_side == 'short'):
                act = 'closing'
            else:
                act = None
        elif sig['macd_div'] < 0:
            if ((sig['macd_div_diff_ema'] < 0)
                    or (self._has_high_volume(df=df_rate)
                        and self._has_high_volatility(df=df_rate))):
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
            'log_str': '{:^44}|'.format(
                '{0} MACD-EMA:{1:>9}{2:>18}'.format(
                    self._parse_granularity(granularity=granularity),
                    '{:.1g}'.format(sig['macd_div']),
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
        ).drop(columns=['mid_ff', 'delta_sec']).assign(
            macd_ema=lambda d:
            d['macd'].ewm(span=self.macd_ema_span, adjust=False).mean()
        )

    def _has_high_volume(self, df):
        e = (
            df['volume'].diff()
            / df.reset_index()['time'].diff().dt.total_seconds()
        ).ewm(span=self.macd_ema_span, adjust=False)
        return (e.mean().iloc[-1] > e.std().iloc[-1])

    def _has_high_volatility(self, df):
        e = (
            self._calculate_hv(df=df).diff()
            / df.reset_index()['time'].diff().dt.total_seconds()
        ).ewm(span=self.macd_ema_span, adjust=False)
        return (e.mean().iloc[-1] > e.std().iloc[-1])

    def _calculate_hv(self, df):
        return np.log(
            df[['ask', 'bid']].mean(axis=1, skipna=True)
        ).diff().rolling(
            window=self.macd_ema_span
        ).std(ddof=0, skipna=True)

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
                    'sharpe_ratio': self._calculate_sharpe_ratio(df=d)
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
    def _calculate_sharpe_ratio(df):
        return df.reset_index().assign(
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).assign(
            adjusted_return=lambda d: (
                (np.exp(np.log(d['mid']).diff()) - 1)
                / d['delta_sec'] * d['delta_sec'].mean()
            )
        )['adjusted_return'].dropna().pipe(
            lambda s: abs(s.mean() / s.std(ddof=1))
        )

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm


class MacdSignalDetector(object):
    def __init__(self, fast_ema_span=12, slow_ema_span=26, macd_ema_span=9,
                 generic_ema_span=9, significance_level=0.01, volume_factor=0,
                 granularity_scorer='Ljung-Box test'):
        assert fast_ema_span < slow_ema_span, 'invalid spans'
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.generic_ema_span = generic_ema_span
        self.significance_level = significance_level
        self.volume_factor = volume_factor
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
            g: self._calculate_sharpe_ratio(
                df_macd=self._calculate_macd(
                    df_rate=d, fast_ema_span=self.fast_ema_span,
                    slow_ema_span=self.slow_ema_span,
                    macd_ema_span=self.macd_ema_span,
                    volume_factor=self.volume_factor
                ),
                span=self.generic_ema_span
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        sig = feature_dict[granularity].iloc[-1].to_dict()
        self.__logger.debug(f'sig: {sig}')
        macd_diff_ci = self._calculate_ci(
            alpha=(1 - self.significance_level), df=(self.macd_ema_span - 1),
            loc=(sig['macd'] - sig['macd_ema']), scale=sig['macd_emse']
        )
        self.__logger.debug(f'macd_diff_ci: {macd_diff_ci}')
        sr_ema_ci = self._calculate_ci(
            alpha=(1 - self.significance_level),
            df=(self.generic_ema_span - 1), loc=sig['sr_ema'],
            scale=sig['sr_emse']
        )
        self.__logger.debug(f'sr_ema_ci: {sr_ema_ci}')
        if sr_ema_ci[0] > 0 and sig['macd'] > sig['macd_ema']:
            act = 'long'
        elif sr_ema_ci[1] < 0 and sig['macd'] < sig['macd_ema']:
            act = 'short'
        elif ((position_side == 'short' and macd_diff_ci[0] > 0
               and sig['sr_ema'] > 0)
              or (position_side == 'long' and macd_diff_ci[1] < 0
                  and sig['sr_ema'] < 0)):
            act = 'closing'
        else:
            act = None
        self.__logger.debug(f'act: {act}')
        return {
            'act': act, 'granularity': granularity, **sig,
            'macd_diff_ci_lower': macd_diff_ci[0],
            'macd_diff_ci_upper': macd_diff_ci[1],
            'sr_ema_ci_lower': sr_ema_ci[0], 'sr_ema_ci_upper': sr_ema_ci[1],
            'log_str': '{0:^7}|{1:^38}|{2:^32}|'.format(
                granularity,
                '\u0394MACD:{0:>10}{1:>18}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    np.array2string(
                        macd_diff_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'SR:{0:>9}{1:>16}'.format(
                    '{:.1g}'.format(sig['sr_ema']),
                    np.array2string(
                        sr_ema_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    @staticmethod
    def _calculate_macd(df_rate, fast_ema_span, slow_ema_span, macd_ema_span,
                        volume_factor=0):
        return df_rate.dropna().assign(
            mid=lambda d: d[['ask', 'bid']].mean(axis=1),
            volume_weight=lambda d:
            np.power(d['volume'], volume_factor).pipe(lambda s: (s / s.mean()))
        ).assign(
            macd=lambda d: (
                (
                    d['mid'].ewm(span=fast_ema_span, adjust=False).mean()
                    - d['mid'].ewm(span=slow_ema_span, adjust=False).mean()
                ) * d['volume_weight']
            )
        ).assign(
            macd_ema=lambda d:
            d['macd'].ewm(span=macd_ema_span, adjust=False).mean(),
            macd_emse=lambda d: np.sqrt(
                d['macd'].ewm(span=macd_ema_span, adjust=False).var(ddof=1)
                / macd_ema_span
            )
        )

    @staticmethod
    def _calculate_sharpe_ratio(df_macd, span):
        return df_macd.assign(
            log_return=lambda d: np.log(d['mid']).diff(),
            delta_sec=lambda d: d.index.to_series().diff().dt.total_seconds()
        ).assign(
            pl_per_sec=lambda d:
            (np.exp(d['log_return'] / d['delta_sec'] * d['volume_weight']) - 1)
        ).assign(
            sharpe_ratio=lambda d: (
                d['pl_per_sec'] * d['bid'] / d['ask']
                / d['pl_per_sec'].rolling(window=span).std(ddof=1)
            )
        ).assign(
            sr_ema=lambda d:
            d['sharpe_ratio'].ewm(span=span, adjust=False).mean(),
            sr_emse=lambda d: np.sqrt(
                d['sharpe_ratio'].ewm(span=span, adjust=False).var(ddof=1)
                / span
            )
        )

    def _select_best_granularity(self, feature_dict):
        if len(feature_dict) == 1:
            granularity = list(feature_dict.keys())[0]
        elif self.granularity_scorer == 'Ljung-Box test':
            best_g = pd.DataFrame([
                {
                    'granularity': g,
                    'pvalue': self._calculate_ljungbox_test_pvalue(
                        x=(d['macd'] - d['macd_ema']).dropna()
                    )
                } for g, d in feature_dict.items()
            ]).pipe(lambda d: d.iloc[d['pvalue'].idxmin()])
            self.__logger.info('pvalue: {}'.format(best_g['pvalue']))
            granularity = best_g['granularity']
        elif self.granularity_scorer == 'Sharpe ratio':
            best_g = pd.DataFrame([
                {
                    'granularity': g,
                    'sharpe_ratio': d['pl_per_sec'].dropna().pipe(
                        lambda s: (s.mean() / s.std(ddof=1))
                    )
                } for g, d in feature_dict.items()
            ]).pipe(lambda d: d.iloc[d['sharpe_ratio'].idxmax()])
            self.__logger.info(
                'sharpe_ratio: {}'.format(best_g['sharpe_ratio'])
            )
            granularity = best_g['granularity']
        else:
            raise ValueError(f'invalid scorer: {self.granularity_scorer}')
        self.__logger.debug(f'granularity: {granularity}')
        return granularity

    @staticmethod
    def _calculate_ljungbox_test_pvalue(x, return_df=True, lags=1, **kwargs):
        return sm.stats.diagnostic.acorr_ljungbox(
            x=x, return_df=return_df, lags=lags, **kwargs
        ).iloc[0]['lb_pvalue']

    @staticmethod
    def _calculate_ci(*args, **kwargs):
        return np.array(scs.t.interval(*args, **kwargs))

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
                 generic_ema_span=9, significance_level=0.01,
                 granularity_scorer='Ljung-Box test'):
        assert fast_ema_span < slow_ema_span, 'invalid spans'
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.generic_ema_span = generic_ema_span
        self.significance_level = significance_level
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
                    macd_ema_span=self.macd_ema_span
                ),
                span=self.generic_ema_span
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        sig = feature_dict[granularity].iloc[-1].to_dict()
        macd_diff_ci = self._calculate_ci(
            alpha=(1 - self.significance_level), df=(self.macd_ema_span - 1),
            loc=(sig['macd'] - sig['macd_ema']), scale=sig['macd_emse']
        )
        emsr_ci = self._calculate_ci(
            alpha=(1 - self.significance_level),
            df=(self.generic_ema_span - 1), loc=sig['sr_ema'],
            scale=sig['sr_emse']
        )
        if macd_diff_ci[0] > 0 and emsr_ci[0] > 0:
            act = 'long'
        elif macd_diff_ci[1] < 0 and emsr_ci[1] < 0:
            act = 'short'
        elif ((position_side == 'short'
               and ((emsr_ci[0] > 0 and sig['macd'] > sig['macd_ema'])
                    or (macd_diff_ci[0] > 0 and sig['sr_ema'] > 0)))
              or (position_side == 'long'
                  and ((emsr_ci[1] < 0 and sig['macd'] < sig['macd_ema'])
                       or (macd_diff_ci[1] < 0 and sig['sr_ema'] < 0)))):
            act = 'closing'
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig,
            'macd_diff_ci_lower': macd_diff_ci[0],
            'macd_diff_ci_upper': macd_diff_ci[1],
            'emsr_ci_lower': emsr_ci[0], 'emsr_ci_upper': emsr_ci[1],
            'log_str': '{0:^7}|{1:^41}|{2:^34}|'.format(
                granularity,
                'MACD-EMA:{0:>10}{1:>18}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    np.array2string(
                        macd_diff_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'EMSR:{0:>9}{1:>16}'.format(
                    '{:.1g}'.format(sig['sr_ema']),
                    np.array2string(
                        emsr_ci, formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    @staticmethod
    def _calculate_macd(df_rate, fast_ema_span, slow_ema_span, macd_ema_span):
        return df_rate[['ask', 'bid']].dropna().assign(
            mid=lambda d: d[['ask', 'bid']].mean(axis=1)
        ).assign(
            macd=lambda d: (
                d['mid'].ewm(span=fast_ema_span, adjust=False).mean()
                - d['mid'].ewm(span=slow_ema_span, adjust=False).mean()
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
            return_rate=lambda d: (np.exp(np.log(d['mid']).diff()) - 1),
            spread_ratio=lambda d: (1 - ((d['ask'] - d['bid']) / d['mid']))
        ).assign(
            sr=lambda d: (
                d['return_rate'] * d['spread_ratio']
                / d['return_rate'].rolling(window=span).std(ddof=1)
            )
        ).assign(
            sr_ema=lambda d: d['sr'].ewm(span=span, adjust=False).mean(),
            sr_emse=lambda d:
            np.sqrt(d['sr'].ewm(span=span, adjust=False).var(ddof=1) / span)
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
                    'sharpe_ratio': d['return_rate'].dropna().pipe(
                        lambda s: s.mean() / s.std(ddof=1)
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

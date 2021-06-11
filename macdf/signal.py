#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class MacdSignalDetector(object):
    def __init__(self, fast_ema_span=12, slow_ema_span=26, macd_ema_span=9,
                 generic_ema_span=9, significance_level=0.01,
                 min_sharpe_ratio=0, granularity_scorer='Ljung-Box test'):
        assert fast_ema_span < slow_ema_span, 'invalid spans'
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.generic_ema_span = generic_ema_span
        self.significance_level = significance_level
        self.min_sharpe_ratio = min_sharpe_ratio
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
            g: self._calculate_adjusted_macd(df=d).pipe(
                lambda f: self._calculate_ewm_sharpe_ratio(
                    df=f, span=self.generic_ema_span,
                    is_short=(f['macd'].iloc[-1] < f['macd_ema'].iloc[-1])
                )
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        df_sig = feature_dict[granularity].assign(
            macd_delta_ema=lambda d: (d['macd'] - d['macd_ema']).diff().ewm(
                span=self.generic_ema_span, adjust=False
            ).mean(),
            emsr_delta_ema=lambda d: d['ewm_sharpe_ratio'].diff().ewm(
                span=self.generic_ema_span, adjust=False
            ).mean()
        )
        self.__logger.info(f'df_sig:{os.linesep}{df_sig}')
        sig = df_sig.iloc[-1]
        lr_ci = df_sig.assign(
            lr=lambda d:
            (np.log(d['mid']).diff() / d['delta_sec'] * d['delta_sec'].mean())
        ).assign(
            lr_ema=lambda d: d['lr'].ewm(
                span=self.generic_ema_span, adjust=False
            ).mean(),
            lr_emstd=lambda d: d['lr'].ewm(
                span=self.generic_ema_span, adjust=False
            ).std(ddof=1)
        ).pipe(
            lambda d: stats.norm.interval(
                alpha=(1 - self.significance_level), loc=d['lr_ema'].iloc[-1],
                scale=d['lr_emstd'].iloc[-1]
            )
        )
        if sig['macd'] > sig['macd_ema']:
            if (sig['macd_delta_ema'] > 0 and sig['emsr_delta_ema'] > 0
                    and (sig['ewm_sharpe_ratio'] >= self.min_sharpe_ratio
                         or lr_ci[0] > 0)):
                act = 'long'
            elif ((position_side == 'long'
                   and ((sig['macd_delta_ema'] < 0
                         and sig['emsr_delta_ema'] < 0)
                        or sig['ewm_sharpe_ratio'] < self.min_sharpe_ratio
                        or lr_ci[1] < 0))
                  or position_side == 'short'):
                act = 'closing'
            else:
                act = None
        elif sig['macd'] < sig['macd_ema']:
            if (sig['macd_delta_ema'] < 0 and sig['emsr_delta_ema'] > 0
                    and (sig['ewm_sharpe_ratio'] >= self.min_sharpe_ratio
                         or lr_ci[1] < 0)):
                act = 'short'
            elif ((position_side == 'short'
                   and ((sig['macd_delta_ema'] > 0
                         and sig['emsr_delta_ema'] < 0)
                        or sig['ewm_sharpe_ratio'] < self.min_sharpe_ratio
                        or lr_ci[0] > 0))
                  or position_side == 'long'):
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig.to_dict(),
            'lr_ci_lower_limit': lr_ci[0], 'lr_ci_upper_limit': lr_ci[1],
            'log_str': '{0:^7}|{1:^31}|{2:^25}|{3:^27}|'.format(
                self._parse_granularity(granularity=granularity),
                'MACD-EMA:{0:>8}{1:>10}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    '[{:.1g}]'.format(sig['macd_delta_ema'])
                ),
                'SR:{0:>8}{1:>10}'.format(
                    '{:.1g}'.format(sig['ewm_sharpe_ratio']),
                    '[{:.1g}]'.format(sig['emsr_delta_ema'])
                ),
                'LRCI:{:>18}'.format(
                    np.array2string(
                        np.array(lr_ci),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    def _calculate_adjusted_macd(self, df):
        return df.dropna(subset=['ask', 'bid']).reset_index().assign(
            mid=lambda d: d[['ask', 'bid']].mean(axis=1),
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).set_index('time').assign(
            macd=lambda d: (
                d['mid'].ewm(span=self.fast_ema_span, adjust=False).mean()
                - d['mid'].ewm(span=self.slow_ema_span, adjust=False).mean()
            ) / d['delta_sec'] * d['delta_sec'].mean()
        ).assign(
            macd_ema=lambda d:
            d['macd'].ewm(span=self.macd_ema_span, adjust=False).mean()
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
            df_g = pd.DataFrame([
                {
                    'granularity': g,
                    'sharpe_ratio': d['return_rate'].dropna().pipe(
                        lambda s: s.mean() / s.std(ddof=1)
                    )
                } for g, d in feature_dict.items()
            ])
            best_g = df_g.pipe(lambda d: d.iloc[d['sharpe_ratio'].idxmax()])
            self.__logger.info(
                'sharpe_ratio: {}'.format(best_g['sharpe_ratio'])
            )
            granularity = best_g['granularity']
        else:
            raise ValueError(f'invalid scorer: {self.granularity_scorer}')
        self.__logger.debug(f'granularity: {granularity}')
        return granularity

    @staticmethod
    def _calculate_ewm_sharpe_ratio(df, span=None, is_short=False):
        return df.assign(
            spread=lambda d: (d['ask'] - d['bid'])
        ).assign(
            return_rate=lambda d: (
                (np.exp(np.log(d['mid']).diff()) - 1) * (-1 if is_short else 1)
                / d['delta_sec'] * d['delta_sec'].mean()
                / d['spread'] * d['spread'].mean()
            )
        ).assign(
            ewm_sharpe_ratio=lambda d: (
                d['return_rate'].ewm(span=span, adjust=False).mean()
                / d['return_rate'].ewm(span=span, adjust=False).std(ddof=1)
            )
        )

    @staticmethod
    def _calculate_ljungbox_test_pvalue(x, return_df=True, lags=1, **kwargs):
        return sm.stats.diagnostic.acorr_ljungbox(
            x=x, return_df=return_df, lags=lags, **kwargs
        ).iloc[0]['lb_pvalue']

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

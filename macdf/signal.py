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
                 trigger_sharpe_ratio=1, signal_count=1,
                 granularity_scorer='Ljung-Box test'):
        assert fast_ema_span < slow_ema_span, 'invalid spans'
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.generic_ema_span = generic_ema_span
        self.significance_level = significance_level
        self.trigger_sharpe_ratio = trigger_sharpe_ratio
        self.signal_count = signal_count
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
            delta_macd_diff=lambda d: (d['macd'] - d['macd_ema']).diff(),
            delta_emsr=lambda d: d['emsr'].diff()
        ).assign(
            delta_macd_diff_ema=lambda d: d['delta_macd_diff'].ewm(
                span=self.generic_ema_span, adjust=False
            ).mean(),
            delta_macd_diff_emvar=lambda d: d['delta_macd_diff'].ewm(
                span=self.generic_ema_span, adjust=False
            ).var(ddof=1),
            delta_emsr_ema=lambda d: d['delta_emsr'].ewm(
                span=self.generic_ema_span, adjust=False
            ).mean(),
            delta_emsr_emvar=lambda d: d['delta_emsr'].ewm(
                span=self.generic_ema_span, adjust=False
            ).var(ddof=1)
        ).tail(self.signal_count)
        sig = df_sig.iloc[-1]
        delta_macd_diff_emci = scs.t.interval(
            alpha=(1 - self.significance_level),
            df=(self.macd_ema_span - 1), loc=sig['delta_macd_diff_ema'],
            scale=np.sqrt(sig['delta_macd_diff_emvar'] / self.macd_ema_span)
        )
        delta_emsr_emci = scs.t.interval(
            alpha=(1 - self.significance_level),
            df=(self.generic_ema_span - 1), loc=sig['delta_emsr_ema'],
            scale=np.sqrt(sig['delta_emsr_emvar'] / self.generic_ema_span)
        )
        if (df_sig['macd'] > df_sig['macd_ema']).all():
            if ((df_sig['emsr'] > 0).all()
                    and sig['delta_macd_diff_ema'] > 0
                    and sig['delta_emsr_ema'] > 0
                    and (delta_macd_diff_emci[0] > 0
                         or delta_emsr_emci[0] > 0
                         or sig['emsr'] >= self.trigger_sharpe_ratio)):
                act = 'long'
            elif ((position_side == 'long' and (df_sig['emsr'] < 0).all()
                   and (delta_macd_diff_emci[1] < 0 or delta_emsr_emci[1] < 0))
                  or position_side == 'short'):
                act = 'closing'
            else:
                act = None
        elif (df_sig['macd'] < df_sig['macd_ema']).all():
            if ((df_sig['emsr'] > 0).all()
                    and sig['delta_macd_diff_ema'] < 0
                    and sig['delta_emsr_ema'] > 0
                    and (delta_macd_diff_emci[1] < 0
                         or delta_emsr_emci[0] > 0
                         or sig['emsr'] >= self.trigger_sharpe_ratio)):
                act = 'short'
            elif ((position_side == 'short' and (df_sig['emsr'] < 0).all()
                   and (delta_macd_diff_emci[0] > 0 or delta_emsr_emci[1] < 0))
                  or position_side == 'long'):
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig.to_dict(),
            'delta_macd_diff_emci_lower': delta_macd_diff_emci[0],
            'delta_macd_diff_emci_upper': delta_macd_diff_emci[1],
            'delta_emsr_emci_lower': delta_emsr_emci[0],
            'delta_emsr_emci_upper': delta_emsr_emci[1],
            'log_str': '{0:^7}|{1:^41}|{2:^35}|'.format(
                self._parse_granularity(granularity=granularity),
                'MACD-EMA:{0:>10}{1:>18}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    np.array2string(
                        np.array(delta_macd_diff_emci),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'EMSR:{0:>9}{1:>17}'.format(
                    '{:.1g}'.format(sig['emsr']),
                    np.array2string(
                        np.array(delta_emsr_emci),
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
            emsr=lambda d: (
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

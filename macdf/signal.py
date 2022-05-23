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
            g: self._calculate_ewm_sharpe_ratio(
                df_macd=self._calculate_adjusted_macd(df_rate=d)
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        macd_diff_emci = self._calculate_emci(
            series=feature_dict[granularity].pipe(
                lambda d: (d['macd'] - d['macd_ema'])
            ),
            span=self.generic_ema_span,
            significance_level=self.significance_level
        )
        emsr_emci = self._calculate_emci(
            series=feature_dict[granularity]['emsr'],
            span=self.generic_ema_span,
            significance_level=self.significance_level
        )
        df_sig = feature_dict[granularity].tail(self.signal_count)
        sig = df_sig.iloc[-1]
        if (df_sig['macd'] > df_sig['macd_ema']).all():
            if ((df_sig['emsr'] > 0).all()
                    and ((macd_diff_emci[0] > 0 and emsr_emci[0] > 0)
                         or sig['emsr'] > self.trigger_sharpe_ratio)):
                act = 'long'
            elif ((position_side == 'short' and (df_sig['emsr'] > 0).all())
                  or (position_side == 'long' and (df_sig['emsr'] < 0).all()
                      and macd_diff_emci[1] < 0 and emsr_emci[1] < 0)):
                act = 'closing'
            else:
                act = None
        elif (df_sig['macd'] < df_sig['macd_ema']).all():
            if ((df_sig['emsr'] < 0).all()
                    and ((macd_diff_emci[1] < 0 and emsr_emci[1] < 0)
                         or sig['emsr'] < -self.trigger_sharpe_ratio)):
                act = 'short'
            elif ((position_side == 'long' and (df_sig['emsr'] < 0).all())
                  or (position_side == 'short' and (df_sig['emsr'] > 0).all()
                      and macd_diff_emci[0] > 0 and emsr_emci[0] > 0)):
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig.to_dict(),
            'macd_diff_emci_lower': macd_diff_emci[0],
            'macd_diff_emci_upper': macd_diff_emci[1],
            'emsr_emci_lower': emsr_emci[0], 'emsr_emci_upper': emsr_emci[1],
            'log_str': '{0:^7}|{1:^41}|{2:^35}|'.format(
                self._parse_granularity(granularity=granularity),
                'MACD-EMA:{0:>10}{1:>18}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    np.array2string(
                        macd_diff_emci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'EMSR:{0:>9}{1:>17}'.format(
                    '{:.1g}'.format(sig['emsr']),
                    np.array2string(
                        emsr_emci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    def _calculate_adjusted_macd(self, df_rate):
        return df_rate.dropna(subset=['ask', 'bid']).reset_index().assign(
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

    @staticmethod
    def _calculate_emci(series, span, significance_level=0.01):
        return np.array(
            scs.t.interval(
                alpha=(1 - significance_level), df=(span - 1),
                loc=series.ewm(span=span, adjust=False).mean().iloc[-1],
                scale=np.sqrt(
                    series.ewm(span=span, adjust=False).var(ddof=1).iloc[-1]
                    / span
                )
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

    def _calculate_ewm_sharpe_ratio(self, df_macd):
        return df_macd.assign(
            spread=lambda d: (d['ask'] - d['bid'])
        ).assign(
            return_rate=lambda d: (
                (np.exp(np.log(d['mid']).diff()) - 1)
                / d['delta_sec'] * d['delta_sec'].mean()
                / d['spread'] * d['spread'].mean()
            )
        ).assign(
            emsr=lambda d: (
                d['return_rate'].ewm(
                    span=self.generic_ema_span, adjust=False
                ).mean() / d['return_rate'].ewm(
                    span=self.generic_ema_span, adjust=False
                ).std(ddof=1)
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

#!/usr/bin/env python

import logging
import os
import warnings
from pprint import pformat

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
            g: self._calculate_adjusted_macd(df=d).pipe(
                lambda f: self._calculate_ewm_sharpe_ratio(
                    df=f, span=self.macd_ema_span,
                    is_short=(f['macd'].iloc[-1] < f['macd_ema'].iloc[-1])
                )
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        df_sig = feature_dict[granularity].assign(
            macd_delta_ema=lambda d: (d['macd'] - d['macd_ema']).diff().ewm(
                span=self.macd_ema_span, adjust=False
            ).mean(),
            emsr_delta_ema=lambda d: d['ewm_sharpe_ratio'].diff().ewm(
                span=self.macd_ema_span, adjust=False
            ).mean()
        )
        self.__logger.info(f'df_sig:{os.linesep}{df_sig}')
        sig = df_sig.iloc[-1]
        if sig['macd'] > sig['macd_ema']:
            if sig['macd_delta_ema'] > 0 and sig['emsr_delta_ema'] > 0:
                act = 'long'
            elif ((position_side == 'long' and sig['macd_delta_ema'] < 0
                   and sig['emsr_delta_ema'] < 0)
                  or position_side == 'short'):
                act = 'closing'
            else:
                act = None
        elif sig['macd'] < sig['macd_ema']:
            if sig['macd_delta_ema'] < 0 and sig['emsr_delta_ema'] > 0:
                act = 'short'
            elif ((position_side == 'short' and sig['macd_delta_ema'] > 0
                   and sig['emsr_delta_ema'] < 0)
                  or position_side == 'long'):
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity, **sig.to_dict(),
            'log_str': '{0:^7}|{1:^41}|{2:^37}|'.format(
                self._parse_granularity(granularity=granularity),
                'MACD-EMA [DELTA]:{0:>9}{1:>11}'.format(
                    '{:.1g}'.format(sig['macd'] - sig['macd_ema']),
                    '[{:.1g}]'.format(sig['macd_delta_ema'])
                ),
                'EMSR [DELTA]:{0:>9}{1:>11}'.format(
                    '{:.1g}'.format(sig['ewm_sharpe_ratio']),
                    '[{:.1g}]'.format(sig['emsr_delta_ema'])
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
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                best_g = pd.DataFrame([
                    {
                        'granularity': g,
                        'pvalue': sm.stats.diagnostic.acorr_ljungbox(
                            x=(d['macd'] - d['macd_ema']).dropna()
                        )[1][0]
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
            return_rate=lambda d: (
                (
                    (1 - d['ask'] / d['bid'].shift(1)) if is_short
                    else (d['bid'] / d['ask'].shift(1) - 1)
                ) / d['delta_sec'] * d['delta_sec'].mean()
            )
        ).assign(
            ewm_sharpe_ratio=lambda d: (
                d['return_rate'].ewm(span=span, adjust=False).mean()
                / d['return_rate'].ewm(span=span, adjust=False).std(ddof=1)
            )
        )

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

#!/usr/bin/env python

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .feature import LogReturnFeature


class MacdSignalDetector(object):
    def __init__(self, feature_type='MID', drop_zero=False, fast_ema_span=12,
                 slow_ema_span=26, macd_ema_span=9,
                 feature_sieve_method='Ljung-Box'):
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.__fs_method = feature_sieve_method
        if feature_type == 'MID':
            self.__lrf = None
            self.__feature_code = feature_type
        else:
            self.__lrf = LogReturnFeature(
                type=feature_type, drop_zero=drop_zero
            )
            self.__feature_code = self.__lrf.code

    def detect(self, history_dict, position_side=None):
        feature_dict = {
            g: self._calculate_adjusted_macd(
                series=self._to_feature(df=d)
            ).tail(self.macd_ema_span * 2)
            for g, d in history_dict.items()
        }
        if len(feature_dict) == 1:
            granularity = list(feature_dict.keys())[0]
        elif self.__fs_method == 'Ljung-Box':
            granularity, pvalue = self._select_best_granularity(
                feature_dict=feature_dict
            )
            self.__logger.debug(f'p-value:\t{pvalue}')
        else:
            raise ValueError(f'invalid method:\t{self.__fs_method}')
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
            'log_str': '{:^48}|'.format(
                '{0} {1} MACD-EMA:{2:>9}{3:>18}'.format(
                    self._parse_granularity(granularity=granularity),
                    self.__feature_code,
                    '{:.1g}'.format(sig['macd_div']),
                    np.array2string(
                        np.array([sig['macd'], sig['macd_ema']]),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    def _calculate_adjusted_macd(self, series):
        return series.to_frame(name='value').reset_index().assign(
            v_ff=lambda d: d['value'].fillna(method='ffill'),
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).assign(
            macd=lambda d: (
                d['v_ff'].ewm(span=self.fast_ema_span, adjust=False).mean()
                - d['v_ff'].ewm(span=self.slow_ema_span, adjust=False).mean()
            ) / d['delta_sec'] * d['delta_sec'].mean()
        ).drop(columns=['v_ff', 'delta_sec']).assign(
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

    @staticmethod
    def _select_best_granularity(feature_dict):
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
        return best_g['granularity'], best_g['pvalue']

    def _to_feature(self, df):
        if 'MID' == self.__feature_code:
            return df[['ask', 'bid']].dropna().mean(axis=1)
        else:
            return self.__lrf.series(df_rate=df).dropna()

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

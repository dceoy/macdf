#!/usr/bin/env python

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tifft.macd import MacdCalculator

from .feature import LogReturnFeature


class MacdSignalDetector(object):
    def __init__(self, feature_type='MID', drop_zero=False, fast_ema_span=12,
                 slow_ema_span=26, macd_ema_span=9,
                 feature_sieve_method='Ljung-Box'):
        self.__logger = logging.getLogger(__name__)
        self.__macdc = MacdCalculator(
            fast_ema_span=fast_ema_span, slow_ema_span=slow_ema_span,
            macd_ema_span=macd_ema_span
        )
        self.__window = macd_ema_span
        self.__fs_method = feature_sieve_method
        if feature_type in {'MID', 'VEL'}:
            self.__lrf = None
            self.__feature_code = feature_type
        else:
            self.__lrf = LogReturnFeature(
                type=feature_type, drop_zero=drop_zero
            )
            self.__feature_code = self.__lrf.code

    def detect(self, history_dict, position_side=None):
        feature_dict = {
            g: self.__macdc.calculate(values=self._to_feature(df=d)).tail(
                self.__window * 2
            ) for g, d in history_dict.items()
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
        df_macd = feature_dict[granularity]
        v_macd_diff = df_macd['macd'] - df_macd['macd_ema']
        if v_macd_diff.tail(2).gt(0).all():
            if (v_macd_diff.tail(self.__window).diff().sum(skipna=True) > 0
                    and (df_macd[['macd', 'macd_ema']].iloc[-1].prod() < 0
                         or self._is_volatile(df=history_dict[granularity]))):
                act = 'long'
            elif position_side == 'short':
                act = 'closing'
            else:
                act = None
        elif v_macd_diff.tail(2).lt(0).all():
            if (v_macd_diff.tail(self.__window).diff().sum(skipna=True) < 0
                    and (df_macd[['macd', 'macd_ema']].iloc[-1].prod() < 0
                         or self._is_volatile(df=history_dict[granularity]))):
                act = 'short'
            elif position_side == 'long':
                act = 'closing'
            else:
                act = None
        else:
            act = None
        return {
            'act': act, 'granularity': granularity,
            'log_str': '{:^48}|'.format(
                '{0} {1} MACD-EMA:{2:>9}{3:>18}'.format(
                    self._parse_granularity(granularity=granularity),
                    self.__feature_code, '{:.1g}'.format(v_macd_diff.iloc[-1]),
                    np.array2string(
                        df_macd[['macd', 'macd_ema']].iloc[-1].values,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    def _is_volatile(self, df):
        return df.assign(
            hv=lambda d: np.log(
                d[['ask', 'bid']].mean(axis=1, skipna=True)
            ).diff().rolling(window=self.__window).std(ddof=0, skipna=True)
        )[['volume', 'hv']].ewm(
            span=self.__window, adjust=False
        ).mean().tail(self.__window).diff().sum(skipna=True).gt(0).all()

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
        elif 'VEL' == self.__feature_code:
            return df[['ask', 'bid', 'time']].dropna().pipe(
                lambda d: (
                    d[['ask', 'bid']].mean(axis=1)
                    / d['time'].diff().dt.total_seconds()
                )
            )
        else:
            return self.__lrf.series(df_rate=df).dropna()

    @staticmethod
    def _parse_granularity(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

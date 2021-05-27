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
        df_macd = feature_dict[granularity].assign(
            macd_div=lambda d: d['macd'] - d['macd_ema']
        ).assign(
            macd_div_ema_diff=lambda d:
            self._ema(series=d['macd_div'], span=self.macd_ema_span).diff()
        ).tail(self.macd_ema_span)
        if df_macd['macd_div'].iloc[-1] > 0:
            if df_macd['macd_div_ema_diff'].sum(skipna=True) > 0:
                act = 'long'
            elif position_side == 'short':
                act = 'closing'
            else:
                act = None
        elif df_macd['macd_div'].iloc[-1] < 0:
            if df_macd['macd_div_ema_diff'].sum(skipna=True) < 0:
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
                    self.__feature_code,
                    '{:.1g}'.format(df_macd['macd_div'].iloc[-1]),
                    np.array2string(
                        df_macd[['macd', 'macd_ema']].iloc[-1].values,
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
                self._ema(series=d['v_ff'], span=self.fast_ema_span)
                - self._ema(series=d['v_ff'], span=self.slow_ema_span)
            ) / d['delta_sec'] * d['delta_sec'].mean()
        ).drop(columns=['v_ff', 'delta_sec']).assign(
            macd_ema=lambda d:
            self._ema(series=d['macd'], span=self.macd_ema_span)
        )

    @staticmethod
    def _ema(series, **kwargs):
        return series.ewm(adjust=False, **kwargs).mean()

    def _has_volatility_or_volume(self, df):
        return df.reset_index().assign(
            delta_sec=lambda d: d['time'].diff().dt.total_seconds()
        ).assign(
            volume_delta_ema=lambda d: self._ema(
                series=(d['volume'] / d['delta_sec']), span=self.macd_ema_span
            ),
            hv_delta_ema=lambda d: self._ema(
                series=(
                    np.log(
                        d[['ask', 'bid']].mean(axis=1, skipna=True)
                    ).diff().rolling(
                        window=self.macd_ema_span
                    ).std(ddof=0, skipna=True) / d['delta_sec']
                ),
                span=self.macd_ema_span
            )
        )[['volume_delta_ema', 'hv_delta_ema']].diff().tail(
            self.macd_ema_span
        ).sum(skipna=True).gt(0).any()

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

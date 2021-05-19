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
            g: self.__macdc.calculate(values=self._to_feature(df=d))
            for g, d in history_dict.items()
        }
        if len(feature_dict) == 1:
            granularity = list(feature_dict.keys())[0]
        elif self.__fs_method == 'Ljung-Box':
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
            granularity = best_g['granularity']
            self.__logger.debug('p-value:\t{}'.format(best_g['pvalue']))
        else:
            raise ValueError(f'invalid method:\t{self.__fs_method}')
        df_macd = feature_dict[granularity]
        macd = df_macd['macd'].iloc[-1]
        macd_ema = df_macd['macd_ema'].iloc[-1]
        is_volatile = (
            history_dict[granularity].assign(
                hv=lambda d: np.log(
                    d[['ask', 'bid']].mean(axis=1, skipna=True)
                ).diff().rolling(
                    window=self.__window
                ).std(ddof=0, skipna=True)
            )[['hv', 'volume']].ewm(
                span=self.__window, adjust=False
            ).mean().tail(self.__window).diff().sum(skipna=True) > 0
        ).all()
        if (macd > macd_ema and (macd > 0 or is_volatile)):
            sig_act = 'long'
        elif (macd < macd_ema and (macd < 0 or is_volatile)):
            sig_act = 'short'
        elif ((position_side == 'long' and macd < macd_ema)
              or (position_side == 'short' and macd > macd_ema)):
            sig_act = 'closing'
        else:
            sig_act = None
        return {
            'act': sig_act, 'granularity': granularity,
            'log_str': '{:^48}|'.format(
                '{0} {1} MACD-EMA:{2:>9}{3:>18}'.format(
                    self._granularity2str(granularity=granularity),
                    self.__feature_code, '{:.1g}'.format(macd - macd_ema),
                    np.array2string(
                        np.array([macd, macd_ema]),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

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
    def _granularity2str(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

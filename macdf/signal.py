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
        self.__ewm_kwargs = {'span': macd_ema_span, 'adjust': False}
        self.__fs_method = feature_sieve_method
        if 'MID' == feature_type:
            self.__lrf = None
            self.__feature_code = 'MID'
        else:
            self.__lrf = LogReturnFeature(
                type=feature_type, drop_zero=drop_zero
            )
            self.__feature_code = self.__lrf.code

    def detect(self, history_dict, position_side=None):
        feature_dict = {
            g: self.__macdc.calculate(
                values=(
                    d[['ask', 'bid']].mean(axis=1)
                    if 'MID' == self.__feature_code else
                    self.__lrf.series(df_rate=d)
                ).dropna()
            ) for g, d in history_dict.items()
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
        vol_score = history_dict[granularity].pipe(
            lambda d:
            np.reciprocal((np.log(d['ask']) - np.log(d['bid'])) * d['volume'])
        ).dropna().ewm(**self.__ewm_kwargs).mean().iloc[-1]
        if macd > macd_ema and (macd > 0 or vol_score > 0):
            sig_act = 'long'
        elif macd < macd_ema and (macd < 0 or vol_score > 0):
            sig_act = 'short'
        elif ((position_side == 'long' and macd < macd_ema)
              or (position_side == 'short' and macd > macd_ema)):
            sig_act = 'closing'
        else:
            sig_act = None
        return {
            'sig_act': sig_act, 'granularity': granularity,
            'sig_log_str': '{:^53}|'.format(
                '{0} {1} MACD/EMA DIFF:{2:>9}{3:>18}'.format(
                    self._granularity2str(granularity=granularity),
                    self.__feature_code, '{:.1g}'.format(macd - macd_ema),
                    np.array2string(
                        np.array([macd, macd_ema]),
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    @staticmethod
    def _granularity2str(granularity='S5'):
        return '{0:0>2}{1:1}'.format(
            int(granularity[1:] if len(granularity) > 1 else 1),
            granularity[0]
        )

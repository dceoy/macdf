#!/usr/bin/env python

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tifft.macd import MacdCalculator

from .feature import LogReturnFeature


class MacdSignalDetector(object):
    def __init__(self, feature_type='LR', drop_zero=False, fast_ema_span=12,
                 slow_ema_span=26, macd_ema_span=9):
        self.__logger = logging.getLogger(__name__)
        self.__ewm_span = macd_ema_span
        self.__macdc = MacdCalculator(
            fast_ema_span=fast_ema_span, slow_ema_span=slow_ema_span,
            macd_ema_span=macd_ema_span
        )
        self.__lrfs = LRFeatureSieve(feature_type=feature_type, drop_zero=True)

    def detect(self, history_dict, pos):
        best_f = self.__lrfs.extract_best_feature(history_dict=history_dict)
        sig = self.__macdc.calculate(
            values=best_f['series']
        ).iloc[-1, ].to_dict()
        macd_score = sig['signal']
        vol_score = best_f['df'].pipe(
            lambda d: np.reciprocal(
                (d['ask'] - d['bid']) * d['volume']
            ).ewm(
                span=self.__ema_span, adjust=False
            ).mean().values[-1]
        )
        if macd_score > 1 or (macd_score > 0 and vol_score > 0):
            sig_act = 'long'
        elif macd_score < -1 or (macd_score < 0 and vol_score < 0):
            sig_act = 'short'
        else:
            sig_act = None
        return {
            'sig_act': sig_act, 'granularity': best_f['granularity'],
            'sig_log_str': '{0:^12}|{1:^30}|'.format(
                '{0:>3}[{1:>3}]'.format(
                    self.__lrfs.lrf.code,
                    self._granularity2str(granularity=best_f['granularity'])
                ),
                'MACD/EMA:{:>17}'.format(
                    np.array2string(
                        np.array([sig['macd'], sig['macd_ema']]),
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


class LRFeatureSieve(object):
    def __init__(self, feature_type='LR', drop_zero=False):
        self.__logger = logging.getLogger(__name__)
        self.lrf = LogReturnFeature(type=feature_type, drop_zero=drop_zero)

    def extract_best_feature(self, history_dict, method='Ljung-Box'):
        feature_dict = {
            g: self.lrf.series(df_rate=d).dropna()
            for g, d in history_dict.items()
        }
        if len(history_dict) == 1:
            granularity = list(history_dict.keys())[0]
        elif method == 'Ljung-Box':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                df_g = pd.DataFrame([
                    {
                        'granularity': g,
                        'pvalue': sm.stats.diagnostic.acorr_ljungbox(x=s)[1][0]
                    } for g, s in feature_dict.items()
                ])
            best_g = df_g.pipe(lambda d: d.iloc[d['pvalue'].idxmin()])
            granularity = best_g['granularity']
            self.__logger.debug('p-value:\t{}'.format(best_g['pvalue']))
        else:
            raise ValueError(f'invalid method name:\t{method}')
        return {
            'series': feature_dict[granularity], 'granularity': granularity,
            'df': history_dict[granularity]
        }

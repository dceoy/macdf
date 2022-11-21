#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .arima import OptimizedArima


class MacdSignalDetector(object):
    def __init__(self, fast_ema_span=12, slow_ema_span=26, macd_ema_span=9,
                 ssr_window=9, significance_level=0.01, volume_factor=0,
                 granularity_scorer='Ljung-Box test'):
        assert fast_ema_span < slow_ema_span, 'invalid spans'
        self.__logger = logging.getLogger(__name__)
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span
        self.macd_ema_span = macd_ema_span
        self.ssr_window = ssr_window
        self.significance_level = significance_level
        self.volume_factor = volume_factor
        granularity_scorers = ['Ljung-Box test', 'Sharpe ratio']
        matched_scorer = [
            s for s in granularity_scorers if (
                granularity_scorer.lower().replace('-', '').replace(' ', '')
                == s.lower().replace('-', '').replace(' ', '')
            )
        ]
        self.min_sample_size = max(
            self.slow_ema_span, self.macd_ema_span, self.ssr_window
        )
        if matched_scorer:
            self.granularity_scorer = matched_scorer[0]
        else:
            raise ValueError(f'invalid scorer: {granularity_scorer}')
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def detect(self, history_dict, position_side=None):
        feature_dict = {
            g: self._calculate_signed_sharpe_ratio(
                df_macd=self._calculate_macd(
                    df_rate=d, fast_ema_span=self.fast_ema_span,
                    slow_ema_span=self.slow_ema_span,
                    macd_ema_span=self.macd_ema_span,
                    volume_factor=self.volume_factor
                ),
                span=self.ssr_window
            ) for g, d in history_dict.items()
        }
        granularity = self._select_best_granularity(feature_dict=feature_dict)
        df_feature = feature_dict[granularity].iloc[2:].reset_index()
        if df_feature.shape[0] < self.min_sample_size:
            delta_macd = ssr = np.nan
            delta_macd_ci = ssr_ci = np.array([np.nan, np.nan])
        else:
            delta_macd, delta_macd_ci = self._calculate_forecast_ci(
                y=df_feature['delta_macd'],
                significance_level=self.significance_level
            )
            ssr, ssr_ci = self._calculate_forecast_ci(
                y=df_feature['signed_sharpe_ratio'],
                significance_level=self.significance_level
            )
        self.__logger.info(
            'delta_macd, delta_macd_ci, ssr, ssr_ci:'
            + f' {delta_macd}, {delta_macd_ci}, {ssr}, {ssr_ci}'
        )
        if ssr_ci[0] > 0 and delta_macd > 0:
            act = 'long'
        elif ssr_ci[1] < 0 and delta_macd < 0:
            act = 'short'
        elif ((position_side == 'short' and delta_macd_ci[0] > 0
               and ssr > 0)
              or (position_side == 'long' and delta_macd_ci[1] < 0
                  and ssr < 0)):
            act = 'closing'
        else:
            act = None
        self.__logger.debug(f'act: {act}')
        return {
            'act': act, 'granularity': granularity,
            'delta_macd_ci_lower': delta_macd_ci[0],
            'delta_macd_ci_upper': delta_macd_ci[1],
            'ssr_ci_lower': ssr_ci[0], 'ssr_ci_upper': ssr_ci[1],
            'log_str': '{0:^7}|{1:^38}|{2:^33}|'.format(
                granularity,
                '\u0394MACD:{0:>10}{1:>18}'.format(
                    '{:.1g}'.format(delta_macd),
                    np.array2string(
                        delta_macd_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                ),
                'SSR:{0:>9}{1:>16}'.format(
                    '{:.1g}'.format(ssr),
                    np.array2string(
                        ssr_ci,
                        formatter={'float_kind': lambda f: f'{f:.1g}'}
                    )
                )
            )
        }

    @staticmethod
    def _calculate_macd(df_rate, fast_ema_span, slow_ema_span, macd_ema_span,
                        volume_factor=0):
        return df_rate.dropna().assign(
            mid=lambda d: d[['ask', 'bid']].mean(axis=1),
            volume_weight=lambda d:
            np.power(d['volume'], volume_factor).pipe(lambda s: (s / s.mean()))
        ).assign(
            macd=lambda d: (
                (
                    d['mid'].ewm(span=fast_ema_span, adjust=False).mean()
                    - d['mid'].ewm(span=slow_ema_span, adjust=False).mean()
                ) * d['volume_weight']
            )
        ).assign(
            macd_ema=lambda d:
            d['macd'].ewm(span=macd_ema_span, adjust=False).mean()
        ).assign(
            delta_macd=lambda d: (d['macd'] - d['macd_ema'])
        )

    @staticmethod
    def _calculate_signed_sharpe_ratio(df_macd, span):
        return df_macd.assign(
            log_return=lambda d: np.log(d['mid']).diff(),
            delta_sec=lambda d: d.index.to_series().diff().dt.total_seconds()
        ).assign(
            log_return_rate=lambda d:
            (d['log_return'] / d['delta_sec'] * d['volume_weight']).fillna(0)
        ).assign(
            pl_ratio=lambda d: (np.exp(d['log_return_rate']) - 1)
        ).assign(
            signed_sharpe_ratio=lambda d: (
                d['pl_ratio'].ewm(span=span, adjust=False).mean()
                * d['bid'] / d['ask']
                / d['pl_ratio'].ewm(span=span, adjust=False).std()
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
            best_g = pd.DataFrame([
                {
                    'granularity': g,
                    'sharpe_ratio':
                    (d['pl_ratio'].mean() / d['pl_ratio'].std())
                } for g, d in feature_dict.items()
            ]).pipe(lambda d: d.iloc[d['sharpe_ratio'].idxmax()])
            self.__logger.info(
                'sharpe_ratio: {}'.format(best_g['sharpe_ratio'])
            )
            granularity = best_g['granularity']
        else:
            raise ValueError(f'invalid scorer: {self.granularity_scorer}')
        self.__logger.debug(f'granularity: {granularity}')
        return granularity

    @staticmethod
    def _calculate_ljungbox_test_pvalue(x, return_df=True, lags=1, **kwargs):
        return sm.stats.diagnostic.acorr_ljungbox(
            x=x, return_df=return_df, lags=lags, **kwargs
        ).iloc[0]['lb_pvalue']

    @staticmethod
    def _calculate_forecast_ci(y, significance_level=0.01, forecast_size=1):
        oa = OptimizedArima(
            y=y, p_range=(1, 1), d_range=(0, 0), q_range=(0, 0),
            model_kw={
                'enforce_stationarity': False, 'enforce_invertibility': False
            }
        )
        forecasts = oa.forecast_frame(
            alpha=significance_level, get_forecast_kw={'steps': forecast_size}
        ).iloc[-1]
        return (
            forecasts['mean'],
            np.array([forecasts['mean_ci_lower'], forecasts['mean_ci_upper']])
        )

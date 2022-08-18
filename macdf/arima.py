#!/usr/bin/env python

import logging
import os
from pprint import pformat
from warnings import catch_warnings, filterwarnings

import numpy as np
from scipy.optimize import brute
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


class OptimizedArima(object):
    def __init__(self, y, test_size=None, p_range=(0, 1), d_range=(0, 1),
                 q_range=(0, 1), ic='aic', model_kw=None, fit_kw=None,
                 **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert 0 < self.test_size < y.size, f'invalid test size: {test_size}'
        self.__logger.info(f'self.test_size: {self.test_size}')
        for k, t in zip('pdq', [p_range, d_range, q_range]):
            assert len(t) == 2 and 0 <= t[0] <= t[1], f'invalid {k} range: {t}'
        self.arima_order_ranges = tuple(
            slice(t[0], t[1] + 1, 1) for t in [p_range, d_range, q_range]
        )
        self.__logger.info(
            f'self.arima_order_ranges: {self.arima_order_ranges}'
        )
        self.ic = ic
        self.model_kw = (model_kw or dict())
        self.fit_kw = (fit_kw or dict())
        self.scipy_optimize_brute_kw = kwargs
        self.arima_order = tuple()
        self.arima = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_arima_order(self):
        if all((s.start == s.stop - 1) for s in self.arima_order_ranges):
            self.arima_order = tuple(s.start for s in self.arima_order_ranges)
        else:
            result = brute(
                func=self._loss, ranges=self.arima_order_ranges,
                args=(
                    self.y.tail(self.test_size), self.model_kw, self.fit_kw,
                    self.ic
                ),
                finish=None, **self.scipy_optimize_brute_kw
            )
            self.__logger.debug(f'result: {result}')
            self.arima_order = tuple(int(i) for i in result)
        self.__logger.info(f'self.arima_order: {self.arima_order}')
        return self.arima_order

    @staticmethod
    def _loss(x, *args):
        logger = logging.getLogger(__name__)
        y, model_kw, fit_kw, ic = args
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            mod = ARIMA(y, order=x, **model_kw)
            with catch_warnings():
                filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    res = mod.fit(**fit_kw)
                except (FloatingPointError, np.linalg.LinAlgError):
                    loss = np.inf
                else:
                    logger.debug(f'res.mle_retvals: {res.mle_retvals}')
                    loss = getattr(res, ic)
        logger.info(f'x, loss: {x}, {loss}')
        return loss

    def fit_parameters(self, y=None, optimize_arima_order=False, model_kw=None,
                       fit_kw=None):
        if optimize_arima_order or not self.arima_order:
            self.optimize_arima_order()
        with (np.errstate(divide='raise', over='raise', under='raise',
                          invalid='raise'),
              catch_warnings()):
            self.arima = ARIMA(
                (self.y if y is None else y), order=self.arima_order,
                **self.model_kw, **(model_kw or dict())
            )
            self.__logger.debug(f'self.arima: {self.arima}')
            with catch_warnings():
                filterwarnings('ignore', category=ConvergenceWarning)
                res = self.arima.fit(**self.fit_kw, **(fit_kw or dict()))
        self.__logger.debug(f'res: {res}')
        self.__logger.info(f'res.mle_retvals: {res.mle_retvals}')
        self.__logger.info('res.summary(): {}'.format(res.summary()))
        return res

    def predict_frame(self, y=None, optimize_arima_order=False, model_kw=None,
                      fit_kw=None, get_prediction_kw=None, **kwargs):
        return self.fit_parameters(
            y=y, optimize_arima_order=optimize_arima_order, model_kw=model_kw,
            fit_kw=fit_kw
        ).get_prediction(**get_prediction_kw).summary_frame(**kwargs)

    def forecast_frame(self, y=None, optimize_arima_order=False, model_kw=None,
                       fit_kw=None, get_forecast_kw=None, **kwargs):
        return self.fit_parameters(
            y=y, optimize_arima_order=optimize_arima_order, model_kw=model_kw,
            fit_kw=fit_kw
        ).get_forecast(**get_forecast_kw).summary_frame(**kwargs)

#!/usr/bin/env python

import logging
import os
from pprint import pformat

import v20


def set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )


def create_api(environment, api_token, stream=False, **kwargs):
    return v20.Context(
        hostname='{0}-fx{1}.oanda.com'.format(
            ('stream' if stream else 'api'), environment
        ),
        token=api_token, **kwargs
    )


def log_response(response, logger=None, expected_status_range=(100, 399)):
    logger = logger or logging.getLogger(__name__)
    res_str = 'response =>' + os.linesep + pformat(vars(response))
    esr = sorted(expected_status_range)
    if esr[0] <= response.status <= esr[-1]:
        logger.debug(res_str)
    else:
        logger.error(res_str)

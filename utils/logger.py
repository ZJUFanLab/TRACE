# -*- encoding: utf-8 -*-
'''
Filename         :logger.py
Desc: code come from https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/core/logger.py
'''
from __future__ import print_function, absolute_import, absolute_import
import pprint
import logging
import warnings
from typing import List
# from common import separator, line
from utils.common import separator, line


class LoggerBase(object):
    """
    Base class for loggers.

    Any custom logger should be derived from this class.
    """

    def log(self, record, step_id, category="train/batch"):
        """
        Log a record.

        Parameters:
            record (dict): dict of any metric
            step_id (int): index of this log step
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        raise NotImplementedError

    def log_config(self, config):
        """
        Log a hyperparameter config.

        Parameters:
            config (dict): hyperparameter config
        """
        raise NotImplementedError



class LoggingLogger(LoggerBase):
    """
    Log outputs with the builtin logging module of Python.

    By default, the logs will be printed to the console. To additionally log outputs to a file,
    add the following lines in the beginning of your code.

    .. code-block: python

        import logging

        format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger = logging.getLogger("")
        logger.addHandler(handler)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log(self, record, step_id, category="train/batch"):
        if category.endswith("batch"):
            self.logger.warning(separator)
        elif category.endswith("epoch"):
            self.logger.warning(line)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("train  %s: %g" % (k, record[k]))
        elif category == "valid/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("valid  %s: %g" % (k, record[k]))
        elif category == "test/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("test   %s: %g" % (k, record[k]))
        else:
            for k in sorted(record.keys()):
                self.logger.warning("%s: %g" % (k, record[k]))

    def log_config(self, config):
        self.logger.warning(pprint.pformat(config))
    
    def summary(self, k, v):
        if isinstance(k, dict) and v is None:
            for key, value in k.items():
                self.logger.warning("Summary - %s: %s" % (key, str(value)))
        else:
            self.logger.warning("Summary - %s: %s" % (k, str(v)))
    def close(self):
        pass

# if __name__ == '__main__':
#     logger = LoggingLogger()
#     # 准备一个包含多个指标的dict
#     metrics = {
#         "accuracy": 0.923,
#         "macro_f1": 0.888,
#         "macro_auc": 0.956,
#         "loss": 0.012
#     }
#     logger.summary(metrics, None)

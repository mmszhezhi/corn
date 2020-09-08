
# coding: utf-8
import os
import time
import pandas as pd
from tsfresh import extract_features
import copy
import numpy as np
import logging
from multiprocessing import Pool
import gc
import hashlib
import pywt

from tsfresh import extract_relevant_features
class FeatureExtract:

    def __init__(self,
                 ts_fc_settings={
            'maximum':
                None,
            'sum_values':
                None,
            'abs_energy':
                None,
            'mean_abs_change':
                None,
            'mean_change':
                None,
            'mean_second_derivative_central':
                None,
            'median':
                None,
            'mean':
                None,
            'length':
                None,
            'standard_deviation':
                None,
            'variance':
                None,
            'skewness':
                None,
            'kurtosis':
                None,
            'absolute_sum_of_changes':
                None,
            'longest_strike_below_mean':
                None,
            'longest_strike_above_mean':
                None,
            'quantile': [{
                'q': 0.1
            }, {
                'q': 0.2
            }, {
                'q': 0.25
            }, {
                'q': 0.3
            }, {
                'q': 0.4
            }, {
                'q': 0.6
            }, {
                'q': 0.7
            }, {
                'q': 0.75
            }, {
                'q': 0.8
            }, {
                'q': 0.9
            }],
            'binned_entropy': [{
                'max_bins': 10
            }],
            'cwt_coefficients': [{
                'widths': (2, 5, 10, 20),
                'coeff': 0,
                'w': 2
            }],
            'ar_coefficient': [{
                'coeff': 0,
                'k': 5
            }, {
                'coeff': 1,
                'k': 5
            }, {
                'coeff': 2,
                'k': 5
            }, {
                'coeff': 3,
                'k': 5
            }, {
                'coeff': 4,
                'k': 5
            }],
            'change_quantiles': [{
                'ql': 0.1,
                'qh': 0.2,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.1,
                'qh': 0.2,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.2,
                'qh': 0.3,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.2,
                'qh': 0.3,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.3,
                'qh': 0.4,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.3,
                'qh': 0.4,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.4,
                'qh': 0.5,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.4,
                'qh': 0.5,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.5,
                'qh': 0.6,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.5,
                'qh': 0.6,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.6,
                'qh': 0.7,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.6,
                'qh': 0.7,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.7,
                'qh': 0.8,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.7,
                'qh': 0.8,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.8,
                'qh': 0.9,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.8,
                'qh': 0.9,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.9,
                'qh': 1.0,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.9,
                'qh': 1.0,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.0,
                'qh': 0.2,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.0,
                'qh': 0.2,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.2,
                'qh': 0.4,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.2,
                'qh': 0.4,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.4,
                'qh': 0.6,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.4,
                'qh': 0.6,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.6,
                'qh': 0.8,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.6,
                'qh': 0.8,
                'isabs': False,
                'f_agg': 'var'
            }, {
                'ql': 0.8,
                'qh': 1.0,
                'isabs': False,
                'f_agg': 'mean'
            }, {
                'ql': 0.8,
                'qh': 1.0,
                'isabs': False,
                'f_agg': 'var'
            }],
            #     'approximate_entropy': [{
            #         'm': 2,
            #         'r': 0.1
            #     }, {
            #         'm': 2,
            #         'r': 0.3
            #     }, {
            #         'm': 2,
            #         'r': 0.5
            #     }, {
            #         'm': 2,
            #         'r': 0.7
            #     }, {
            #         'm': 2,
            #         'r': 0.9
            #     }],
            'linear_trend': [{
                'attr': 'pvalue'
            }, {
                'attr': 'rvalue'
            }, {
                'attr': 'intercept'
            }, {
                'attr': 'slope'
            }, {
                'attr': 'stderr'
            }],
            'energy_ratio_by_chunks': [{"num_segments": 10, "segment_focus": i} for i in range(10)] +
                                      [{"num_segments": 5, "segment_focus": i} for i in range(5)] +
                                      [{"num_segments": 20, "segment_focus": i} for i in range(20)],
            "number_peaks": [{"n": n} for n in [2, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]],
            "c3": [{"lag": lag} for lag in [2, 5, 8, 10, 15, 20, 25, 30]],
            "cid_ce": [{"normalize": True}, {"normalize": False}],
            "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),

        }, feature_name=None,
               rolling=0, n_jobs=4, wavelet=20):

        self.ts_fc_settings = ts_fc_settings
        self.feature_name = feature_name
        self.features = None  # 特征数据


    # def process(self):
    #     extract_relevant_features(timeseries_container,default_fc_parameters=self.ts_fc_settings)





if __name__ == '__main__':
    import joblib

    arr = extract_features({"data":np.random.normal(1,2,100)})
    print(arr)


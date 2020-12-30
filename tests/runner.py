#!/usr/bin/env python
# coding: utf-8

import numpy as np
import shap
from py3ai import io
import argparse
import os
import math
import pandas as pd
import tempfile
from sklearn.externals import joblib
import concurrent.futures
# TODO - currently the custom docker image doesn't have this package - not really needed though
#from azureml.core.run import Run
import logging as log
import sys

#run = Run.get_context()
log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log.info('Initialising...')


def get_batch_rows(row_count, batch_count, index):
    batch_size = math.ceil(row_count/batch_count)
    if index * batch_size >= row_count:
        return None
    return (index * batch_size, min((index+1)*batch_size, row_count))

def batch_rows(top_range, num_batches, index, model_path, data_path, output_path):
    batch_size = math.ceil((top_range[1]-top_range[0])/num_batches)
    for ndx in range(top_range[0], top_range[1], batch_size):
        yield (ndx, min(ndx + batch_size, top_range[1])), index, model_path, data_path, output_path

def run_shap(inputs):
    (range, index, model_path, data_path, output_path) = inputs
    log.info(f'Running shap for index: {index}, range: {range[0]} : {range[1]}, data_path: {data_path}, model_path {model_path}, output_path: {output_path}')
    train_x = io.load(os.path.join(data_path, 'X0_1.mat'))[range[0]:range[1],:]
    # cols = pd.read_csv(os.path.join(data_path, 'feature_sets.csv'), index_col=0)
    X = pd.DataFrame(train_x).fillna(0.5)
    log.info('Loading model')
    model = joblib.load(os.path.join(model_path, 'treebagger_2_0_52_53_20.pkl'))
    explainer = shap.TreeExplainer(model)
    log.info('Running shap calcs')
    shap_values = explainer.shap_values(X)
    log.info('Writing results')
    output_file = os.path.join(tempfile.gettempdir(), f'shap_values_{range[0]}.h5')
    io.save(output_file, 'shaps', shap_values)
    log.info('Done')
    return output_file

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-data_path', '--data_path', help='Data directory where the dataset is located', required=True)
    # parser.add_argument('-output_path', '--output_path', help='Output path for shap values', required=True)
    # parser.add_argument('-model_path', '--model_path', help='Output path for models', required=True)
    # parser.add_argument('-index', '--index', help='Index of X data to process', required=True, type=int)
    # parser.add_argument('-batch_count', '--batch_count', help='Total number of batches', required=True, type=int)
    # parser.add_argument('-workers', '--workers', help='Number of worker processes to use', required=False, type=int, default=4)
    # args = vars(parser.parse_args())
    # data_path = args['data_path']
    # output_path = args['output_path']
    # model_path = args['model_path']
    # index = args['index']
    # batch_count = args['batch_count']
    # workers = args['workers']

    data_path = '/S3/alba-data/processed/Live/int_live_20200501'
    output_path = '/S3/alba-data/scratch/shap'
    model_path = '/S3/alba-experiments/experiments/TreeBaggerBacktest_backtest_20191108/models/treebagger'
    workers = 8
    batch_count = 160

    # each worker commits up to 6GB RAM initially, so running 4x on a 28GB DS4_v2 is ok, 6x crashes
    # tried loading model and x data before forking (linux) but workers all hang
    log.info('Loading X data')
    train_x = io.load(os.path.join(data_path, 'X0_1.mat'))

    for index in range(batch_count):
        top_range = get_batch_rows(train_x.shape[0], batch_count, index)
        if top_range is None:
            log.info('Nothing to do')
            continue
        log.info('Running jobs on process pool')
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for batch in executor.map(run_shap, batch_rows(top_range, workers, index, model_path, data_path, output_path)):
                results.append(batch)
        log.info('Aggregating results')
        shaps = []
        for s in results:
            shaps.append(io.load(s))

        log.info('Saving aggregated result')
        shap_values = np.concatenate(shaps, axis=0)
        io.save(os.path.join(output_path, f'shap_{index}.h5'), 'shap_values', shap_values)
    log.info('Completed')
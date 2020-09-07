import numpy as np
import argparse
import os
import torch
import itertools
import pandas
import sys
from datetime import datetime

from meta_learn.GPR_meta_svgd import GPRegressionMetaLearnedSVGD
from meta_learn.GPR_meta_vi import GPRegressionMetaLearnedVI
from meta_learn.GPR_meta_mll import GPRegressionMetaLearned
from meta_learn.NPR_meta import NPRegressionMetaLearned
from meta_learn.MAML import MAMLRegression

DATASETS = [
    '%s_%i' % (dataset, n_tasks) for n_tasks in [5, 10, 20, 40, 80, 160, 320]
    for dataset in ['cauchy', 'sin']
]

MODEL_SEEDS = [22, 23, 24, 25, 26]

LAYER_SIZES = [32, 32, 32, 32]


def fit_eval_meta_algo(param_dict):
    if param_dict['mean_nn_layers']:
        param_dict['mean_nn_layers'] = [int(n) for n in param_dict['mean_nn_layers'].split(',')]
    if param_dict['kernel_nn_layers']:
        param_dict['kernel_nn_layers'] = [int(n) for n in param_dict['kernel_nn_layers'].split(',')]
    meta_learner = param_dict["meta_learner"]
    dataset = param_dict.pop("dataset")

    ALGO_MAP = {
        'gpr_meta_mll': GPRegressionMetaLearned,
        'gpr_meta_vi': GPRegressionMetaLearnedVI,
        'gpr_meta_svgd': GPRegressionMetaLearnedSVGD,
        'maml': MAMLRegression,
        'neural_process': NPRegressionMetaLearned,
    }
    meta_learner_cls = ALGO_MAP[meta_learner]

    # 1) Generate Data
    from experiments.data_sim import provide_data
    data_train, _, data_test = provide_data(dataset, 25)

    # 2) Fit model (meta-learning/meta-training)
    model = meta_learner_cls(data_train, **param_dict)
    EVAL_EVERY = 100
    model.meta_fit(data_test, log_period=EVAL_EVERY)

    # 3) evaluate model (meta-testing)
    if meta_learner == 'neural_process':
        eval_result = model.eval_datasets(data_test, flatten_y=False)
    else:
        eval_result = model.eval_datasets(data_test)
    ll, rmse, calib_err = eval_result

def main(args):
    param_dict = vars(args)
    fit_eval_meta_algo(param_dict)
    param_configs = [
        {
            'meta_learner': 'gpr_meta_mll',
            'dataset': DATASETS,
            'seed': MODEL_SEEDS,
            'covar_module': ['NN'],
            'mean_module': 'NN',
            'num_iter_fit': 40000,
            'weight_decay': 0.0,
            'task_batch_size': [4],
            'lr_decay': [0.97],
            'lr_params': [5e-3, 1e-3, 5e-4],
            'mean_nn_layers': [LAYER_SIZES],
            'kernel_nn_layers': [LAYER_SIZES],
        },
        {
            'meta_learner': 'maml',
            'dataset': DATASETS,
            'seed': MODEL_SEEDS,
            'num_iter_fit': 40000,
            'task_batch_size': 4,
            'lr_inner': [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'layer_sizes': [LAYER_SIZES],
        },
        {
            'meta_learner': 'neural_process',
            'dataset': DATASETS,
            'seed': MODEL_SEEDS,
            'num_iter_fit': 40000,
            'task_batch_size': 4,
            'lr_decay': 0.97,
            'lr_params': 1e-3,
            'r_dim': [32, 64, 124],
            'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 4e-1, 8e-1]
        },
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run meta mll hyper-parameter search.')
    parser.add_argument('--meta_learner', type=str, default='neural_process')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--random_seed', type=int, help='Seed for the model.')
    parser.add_argument('--num_iter_fit', type=int)
    parser.add_argument('--task_batch_size', type=int)
    parser.add_argument('--r_dim', type=int)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--lr_params', type=float)
    parser.add_argument('--weight_decay', type=float)
    # Just for Meta-GPR
    parser.add_argument('--mean_module', type=str)
    parser.add_argument('--covar_module', type=str)
    parser.add_argument('--mean_nn_layers', type=str)
    parser.add_argument('--kernel_nn_layers', type=str)

    args = parser.parse_args()

    print('Running', os.path.abspath(__file__), '\n')
    print('\n')

    main(args)
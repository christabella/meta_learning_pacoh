import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

from experiments.data_sim import SinusoidNonstationaryDataset, MNISTRegressionDataset, \
    PhysionetDataset, GPFunctionsDataset, SinusoidDataset, CauchyDataset, provide_data

DATASETS = [
    '%s_%i' % (dataset, n_tasks) for n_tasks in [5, 10, 20, 40, 80, 160, 320]
    for dataset in ['cauchy', 'sin']
]
DATA_SEED = 25
MODEL_SEEDS = [22, 23, 24, 25, 26]

LAYER_SIZES = [32, 32, 32, 32]


def fit_eval_meta_algo(args):
    param_dict = vars(args)
    if param_dict['mean_nn_layers']:
        param_dict['mean_nn_layers'] = [int(n) for n in param_dict['mean_nn_layers'].split(',')]
    if param_dict['kernel_nn_layers']:
        param_dict['kernel_nn_layers'] = [int(n) for n in param_dict['kernel_nn_layers'].split(',')]
    meta_learner = param_dict["meta_learner"]

    ALGO_MAP = {
        'gpr_meta_mll': GPRegressionMetaLearned,
        'gpr_meta_vi': GPRegressionMetaLearnedVI,
        'gpr_meta_svgd': GPRegressionMetaLearnedSVGD,
        'maml': MAMLRegression,
        'neural_process': NPRegressionMetaLearned,
        'conditional_neural_process': NPRegressionMetaLearned,
        'attentive_conditional_neural_process': NPRegressionMetaLearned,
        'conv_conditional_neural_process': NPRegressionMetaLearned,
        'neural_process': NPRegressionMetaLearned,
        'attentive_neural_process': NPRegressionMetaLearned,
    }
    meta_learner_cls = ALGO_MAP[meta_learner]
    param_dict['use_attention'], param_dict['is_conditional'] = False, False
    if "attentive" in meta_learner:
        param_dict['use_attention'] = True
    if "conditional" in meta_learner:
        param_dict['is_conditional'] = True
    if "conv" in meta_learner:
        param_dict['is_conv'] = True
    dataset = None
    # 1) Generate Data
    if args.dataset == 'sin-nonstat':
        dataset = SinusoidNonstationaryDataset(random_state=np.random.RandomState(DATA_SEED + 1))
    elif args.dataset == 'sin':
        dataset = SinusoidDataset(random_state=np.random.RandomState(DATA_SEED + 1))
    elif args.dataset == 'cauchy':
        dataset = CauchyDataset(random_state=np.random.RandomState(DATA_SEED + 1), ndim_x=1)
    elif args.dataset == 'mnist':
        dataset = MNISTRegressionDataset(random_state=np.random.RandomState(DATA_SEED + 1), context_mask=args.context_mask)
        param_dict['image_size'] = 28
    elif args.dataset == 'physionet':
        dataset = PhysionetDataset(random_state=np.random.RandomState(DATA_SEED + 1))
    elif args.dataset == 'gp-funcs':
        dataset = GPFunctionsDataset(random_state=np.random.RandomState(DATA_SEED + 1))
    # If NP, split meta-test into context-target, else put everything into "context"self.
    # Actually even if NP, put *everything* into the meta_train, since within
    # NPR_meta we further do the splitting based on context_split_ratio.
    data_train = dataset.generate_meta_train_data(
        n_tasks=args.n_train_tasks, n_samples=args.n_samples_per_task
    )
    print(f"Data_train: {[x.shape for x in data_train[0]]}")
    data_test = dataset.generate_meta_test_data(
        n_tasks=args.n_test_tasks,
        n_samples_context=args.n_test_context_samples,
        # The "extra target"
        n_samples_test=args.n_samples_per_task - args.n_test_context_samples,
    )
    print(f"Data_test: {[x.shape for x in data_test[0]]}")

    # 2) Fit model (meta-learning/meta-training)
    model = meta_learner_cls(data_train, **param_dict)
    EVAL_EVERY = args.num_iter_fit / 20  # Tensorboard can only show 10 images anyway...
    model.meta_fit(data_test,  # Pass in data_test as validation since we don't
                   # really do anything with it e.g. early stopping...
                   log_period=EVAL_EVERY)

    # 3) evaluate model (meta-testing)
    if meta_learner == 'neural_process':
        eval_result = model.eval_datasets(data_test, flatten_y=False)
    else:
        eval_result = model.eval_datasets(data_test)
    ll, rmse, calib_err = eval_result

def main(args):
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
    parser.add_argument('--context_mask', type=str)
    parser.add_argument('--random_seed', type=int, help='Seed for the model.')
    parser.add_argument('--num_iter_fit', type=int)
    parser.add_argument('--task_batch_size', type=int)
    parser.add_argument('--n_train_tasks', type=int)
    parser.add_argument('--n_samples_per_task', type=int, default=50)
    parser.add_argument('--n_test_tasks', type=int, default=200)
    parser.add_argument('--n_test_context_samples', type=int, default=5)
    parser.add_argument('--r_dim', type=int)
    parser.add_argument('--h_dim', type=int)
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
    try:
        fit_eval_meta_algo(args)
    except:
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()

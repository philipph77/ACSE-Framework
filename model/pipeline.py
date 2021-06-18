from sklearn import preprocessing
from run_on_dataset import run_on_dataset
import time
import numpy as np
from decimal import Decimal
from helper_funcs import generate_name, generate_dataset, assert_hyperparameters
from os import makedirs, system
import preprocessing_funcs

def pipeline(dataset_path, dataset_list, architecture_list, dir_name, max_epochs, min_lam, max_lam, step_lam, runs_per_config, version, suppress_hyperparams_check=False, preprocessing_func=None):
    # dataset_path: (str) path to your preprocessed datasets
    # dataset_list: (list of str) List of datasets you want to use, supported values: SEED, SEED_IV, DEAP, DREAMER
    # architecture_list: (list of str)architectures you want to train, supported values: DeepConvNet, EEGNet, ShallowConvNet, MLP
    # max_epochs: (int) maximal number of epochs for the training
    # min_lam: (float) minimal lambda value you want to use
    # max_lam: (float) maximal lambda value you want to use
    # step_lam: (float) step for lambda values
    # runs_per_config: (int) number of runs you want to train per configuration
    # version: (str) will be appended to the file name for tracability
    # suppress_hyperparams_check: (Bool) set to True if you don't want to be asked if the hyperparameters are correct before each run
    # preprocessing_func: (function) the preprocessing you want to apply. You can pass one of the functions specified in preprocessing_funcs.py
    
    if not suppress_hyperparams_check:
        assert_hyperparameters(dataset_path, dataset_list, architecture_list, dir_name, max_epochs, min_lam, max_lam, step_lam, version)
    
    lam_list = np.arange(Decimal(str(min_lam)), Decimal(str(max_lam)), Decimal(str(step_lam)))
    train_set, val_set, test_set = generate_dataset(dataset_path, dataset_list)

    if not (preprocessing_func==None):
        train_set, val_set, test_set = preprocessing_func(train_set, val_set, test_set)

    makedirs('logs/'+dir_name)

    for architecture in architecture_list:
        for lam in lam_list:
            for run_idx in range(runs_per_config):
                run_name = generate_name(architecture, lam, dataset_list, version, run_idx)
                print(run_name)
                log_path = 'logs/' + dir_name + '/' + run_name
                makedirs(log_path)
                run_on_dataset(train_set, val_set, test_set, run_name, log_path, ARCHITECTURE=architecture, EPOCHS=max_epochs, LAMBDA=lam)

if __name__ == '__main__':
    # Add your code to run here
    pipeline('PATH_TO_YOUR_DATASETS', ['SEED', 'SEED_IV', 'DEAP', 'DREAMER'], ['DeepConvNet'], 'DCN-0-1111-ds-bs2-monitoring', 500, 0.00, 0.05, 0.05, 5, 'bs2', suppress_hyperparams_check=True)
import numpy as np
import time
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def assert_hyperparameters(dataset_path, dataset_list, architecture_list, dir_name, max_epochs, min_lam, max_lam, step_lam, version):
    print('###########################################################################')
    print('Starting the pipeline with the following hyperparameters: ')
    print('dataset_path: %s' %dataset_path)
    print('dataset_list: %s' %dataset_list)
    print('architecture_list: %s' %architecture_list)
    print('dir_name: %s' %dir_name)
    print('max_epochs: %s' %str(max_epochs))
    print('min_lam: %s' % str(min_lam))
    print('max_lam: %s' %str(max_lam))
    print('step_lam: %s' %str(step_lam))
    print('version: %s' %str(version))
    print('###########################################################################')
    user_input = input("Do you want to start the pipeline with these hyperparameters? (Yes/No)")
    while(user_input.lower() not in ['y', 'yes', 'j', 'ja', 'n', 'no', 'nein']):
        print("Unrecognized Answer, please use (Yes/No)")
        user_input = input("Do you want to start the pipeline with these hyperparameters? (Yes/No)")
    assert(user_input.lower() in ['y', 'yes', 'j', 'ja'])


def generate_name(architecture, lam, dataset_list, version, run_idx):
    timestamp = int(time.time())
    name = ''
    if architecture == 'DeepConvNet' or architecture == 'DCN':
        name = name + 'DCN'
    elif architecture == 'EEGNet':
        name = name + 'EEG'
    elif architecture == 'ShallowConvNet' or architecture == 'SCN':
        name = name + 'SCN'
    elif architecture == 'MLP':
        name = name + 'MLP'
    else:
        try:
            raise NotImplementedError
        finally:
            print("Unknown Architecture used")
    
    name = name + '-'
    name = name + str(lam)
    
    name = name + '-'
    for ds_name in ['SEED', 'SEED_IV', 'DEAP', 'DREAMER']:
        if ds_name in dataset_list:
            name = name + '1'
        else:
            name = name + '0'
    
    name = name + '-'
    if True:
        name = name+'ds'
    else:
        raise 'Using unstratified Dataset is not supported'
    
    name = name+'-'
    name = name+'v'+str(version)
    
    name = name+'-'
    name = name+str(run_idx)

    name = name+'-'
    name = name + str(timestamp)
    return name

def get_dataset_list_from_setup_name(setup_name):
    dataset_digits = setup_name.split('-')[2]
    dataset_list = list()
    if dataset_digits[0] == '1':
        dataset_list.append('SEED')
    if dataset_digits[1] == '1':
        dataset_list.append('SEED_IV')
    if dataset_digits[2] == '1':
        dataset_list.append('DEAP')
    if dataset_digits[3] == '1':
        dataset_list.append('DREAMER')
    return dataset_list

def generate_dataset(dataset_path, dataset_list):

    X_train = np.zeros((0,14,256))
    X_val = np.zeros((0,14,256))
    X_test = np.zeros((0,14,256))
    Y_train = np.zeros((0,),dtype=int)
    Y_val = np.zeros((0,),dtype=int)
    Y_test = np.zeros((0,),dtype=int)
    S_train = np.zeros((0,),dtype=int)
    S_val = np.zeros((0,),dtype=int)
    S_test = np.zeros((0,),dtype=int)
    P_train = np.zeros((0,),dtype=int)
    P_val = np.zeros((0,),dtype=int)
    P_test = np.zeros((0,),dtype=int)
    for idx, dataset_name in enumerate(tqdm(dataset_list)):
        filename=dataset_path+ 'bs2_'+dataset_name+'.npz'
        dataset = np.load(filename)
        if idx==0:
            X_train = dataset['X_train']
            X_val = dataset['X_val']
            X_test =dataset['X_test']
            Y_train = dataset['Y_train']
            Y_val = dataset['Y_val']
            Y_test = dataset['Y_test']
            S_train = dataset['S_train']+100*(idx+1)
            S_val = dataset['S_val']+100*(idx+1)
            S_test = dataset['S_test']+100*(idx+1)
            P_train = dataset['P_train']
            P_val = dataset['P_val']
            P_test = dataset['P_test']
        else:
            X_train = np.vstack([X_train,dataset['X_train']])
            X_val = np.vstack([X_val,dataset['X_val']])
            X_test =np.vstack([X_test,dataset['X_test']])
            Y_train = np.concatenate((Y_train,dataset['Y_train']),axis=0)
            Y_val = np.concatenate((Y_val,dataset['Y_val']),axis=0)
            Y_test = np.concatenate((Y_test,dataset['Y_test']),axis=0)
            S_train = np.concatenate((S_train,dataset['S_train']+100*(idx+1)),axis=0)
            S_val = np.concatenate((S_val,dataset['S_val']+100*(idx+1)),axis=0)
            S_test = np.concatenate((S_test,dataset['S_test']+100*(idx+1)),axis=0)
            P_train = np.concatenate((P_train,dataset['P_train']),axis=0)
            P_val = np.concatenate((P_val,dataset['P_val']),axis=0)
            P_test = np.concatenate((P_test,dataset['P_test']),axis=0)
    
    train_set = shuffle(X_train, Y_train, S_train, P_train, random_state=7)
    val_set = shuffle(X_val, Y_val, S_val, P_val, random_state=7)
    test_set = [X_test, Y_test, S_test, P_test]

    return train_set, val_set, test_set

def encode_dataset(train_set, val_set, test_set):
    X_train = train_set[0]
    Y_train = train_set[1]
    P_train = train_set[3]
    X_val = val_set[0]
    Y_val = val_set[1]
    P_val = val_set[3]
    X_test = test_set[0]
    Y_test = test_set[1]
    P_test = test_set[3]
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    ohe = OneHotEncoder(categories='auto', sparse=False)
    Y_train = ohe.fit_transform(Y_train.reshape(-1,1))
    Y_val = ohe.fit_transform(Y_val.reshape(-1,1))
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1))
    ohe = OneHotEncoder(categories='auto', sparse=False)
    P_train = ohe.fit_transform(P_train.reshape(-1,1))
    P_val = ohe.fit_transform(P_val.reshape(-1,1))
    P_test = ohe.fit_transform(P_test.reshape(-1,1))

    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test


def get_run_paths(setup):
    from jellyfish import levenshtein_distance
    import os
    if 'bs' in setup:
        folder = os.path.join('logs', setup)
        files = os.listdir(folder)
        runs = [run  for run in files if levenshtein_distance(run.replace('-0.0-', '-0-')[:17], setup)<10]
        runs = [os.path.join('logs', setup, run) for run in runs]
    elif 'adv' in setup:
        if '-X-' in setup:
            raise NotImplementedError
        else:
            setup_lam = setup.split('-')[1]
            setup_folder = setup.replace(setup_lam, 'X')
            sub_folders = os.listdir(os.path.join('logs', setup_folder))
            runs = [run for run in sub_folders if levenshtein_distance(run[:18],setup[:18])==1]
            runs = [os.path.join('logs', setup_folder, run) for run in runs]

    assert len(runs)==5

    return runs

def count_initial_epochs(setup):
    import pandas as pd
    import os

    train_logs = pd.read_csv(os.path.join(setup, 'train.csv'))
    return len(train_logs)+1
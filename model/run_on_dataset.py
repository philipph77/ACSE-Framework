def run_on_dataset(train_set, val_set, test_set, run_name, log_path, ARCHITECTURE='DeepConvNet', EPOCHS=500, LAMBDA=0.1):

    import numpy as np
    from AdversarialCNN import AdversarialCNN
    from sklearn.preprocessing import OneHotEncoder
    import keras
    import time

    timestamp = int(time.time())

    # Prepare Dataset
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

    # Generate Model
    model = AdversarialCNN(
        chans = X_train.shape[1], # EEG channels
        samples = X_train.shape[2], # number of sampling points per timeseries
        n_output = Y_train.shape[1], # number of label classses
        n_nuisance = P_train.shape[1], # number of domains (possible values of the nuisance variable)
        architecture=ARCHITECTURE,
        adversarial=True,
        lam=float(LAMBDA))

    # Safe run configuration to file
    with open('run_configs/configs.csv', 'a') as f:
        f.write(
            str(timestamp) + ',' +
            run_name + '\n'
        )

    # Train Model
    model.train(
        [X_train, Y_train, P_train],
        [X_val, Y_val, P_val],
        log_path,
        early_stopping_after_epochs=50,
        epochs = EPOCHS,
        run_name = run_name
        )

    # Save Model
    model.enc.save(log_path+'/enc.h5')
    model.cla.save(log_path+'/cla.h5')
    model.adv.save(log_path+'/adv.h5')

    # Test Model
    test_log = model.acnn.test_on_batch(X_test, [Y_test, P_test])
    acc = test_log[3]
    print("acc: %4.2f" % (100.0*test_log[3]))
    with open(log_path+'/test.csv', 'a') as f:
                f.write(str(test_log[0]) + ',' + str(test_log[1]) + ',' +
                        str(100*test_log[3]) + ',' + str(test_log[2]) + ',' + str(100*test_log[4]) + '\n')
    with open(log_path+'/../test.csv', 'a') as f:
                f.write(run_name + ',' + str(test_log[0]) + ',' + str(test_log[1]) + ',' +
                        str(100*test_log[3]) + ',' + str(test_log[2]) + ',' + str(100*test_log[4]) + '\n')

if __name__ == '__main__':
    try:
        import numpy as np
        from AdversarialCNN import AdversarialCNN
        from sklearn.preprocessing import OneHotEncoder
        import keras
        import time
    except:
        print("Not all necessary packages could be imported - Please check your python environment")
        print("Necessary packages: numpy, sklearn, keras, time")

    print("Do not run this file, run pipeline.py instead")

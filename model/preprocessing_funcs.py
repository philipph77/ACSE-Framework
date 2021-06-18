def psd_transformation(trainset, valset, testset, sampling_freq=128, relative_values=True):
    # calculates PSD Features for delta, alpha, beta and gamma band
    # trainset: training data
    # valset: validation data
    # testset: test data
    # sampling_freq: the sampling frequency of the timeseries
    #relative_values: if True, than relative values will be returned
    def psd_transformation_helper(subset, sampling_freq=128, relative_values= True):
        print('Calculating PSD Features')

        import numpy as np
        from scipy.signal import periodogram
        from scipy.integrate import simps
        from tqdm import trange

        X = subset[0]
        
        delta_freqs = [1, 4]
        theta_freqs = [4,7]
        alpha_freqs = [8,13]
        beta_freqs = [13,30]
        gamma_freqs = [30, np.inf]
        
        X_psd = np.zeros((X.shape[0],X.shape[1],5))
        for window in trange(X.shape[0]):
            for channel in range(X.shape[1]):
                freqs, psd = periodogram(X[window, channel,:], sampling_freq)
                freq_res = freqs[1] - freqs[0]
                idx_delta = np.logical_and(freqs >= delta_freqs[0], freqs <= delta_freqs[1])
                idx_theta = np.logical_and(freqs >= theta_freqs[0], freqs <= theta_freqs[1])
                idx_alpha = np.logical_and(freqs >= alpha_freqs[0], freqs <= alpha_freqs[1])
                idx_beta = np.logical_and(freqs >= beta_freqs[0], freqs <= beta_freqs[1])
                idx_gamma = np.logical_and(freqs >= gamma_freqs[0], freqs <= gamma_freqs[1])
                
                if relative_values:
                    total_power = simps(psd, dx=freq_res)
                else:
                    total_power = 1
                
                X_psd[window,channel,0] = simps(psd[idx_delta], dx=freq_res) / total_power
                X_psd[window,channel,1] = simps(psd[idx_theta], dx=freq_res) / total_power
                X_psd[window,channel,2] = simps(psd[idx_alpha], dx=freq_res) / total_power
                X_psd[window,channel,3] =simps(psd[idx_beta], dx=freq_res) / total_power
                X_psd[window,channel,4] =simps(psd[idx_gamma], dx=freq_res) / total_power

        subset[0] = X_psd
                    
        return subset
    
    trainset = psd_transformation_helper(trainset, sampling_freq=sampling_freq, relative_values= relative_values)
    valset =  psd_transformation_helper(valset, sampling_freq=sampling_freq, relative_values= relative_values)
    testset = psd_transformation_helper(testset, sampling_freq=sampling_freq, relative_values= relative_values)
    return trainset, valset, testset

def pick_binary(trainset, valset, testset):
    # return only the samples, that have negative or positive label (filters out the ones with neutral label)
    # trainset: training data
    # valset: validation data
    # testset: test data
    def pick_binary_helper(subset):
        import numpy as np
        X = subset[0]
        Y = subset[1]
        S = subset[2]
        P = subset[3]

        pos_idx = np.argwhere(Y==1)
        neg_idx = np.argwhere(Y==-1)
        indices = np.concatenate((pos_idx, neg_idx), axis=0).squeeze()

        X_bin = np.take(X,indices,0)
        Y_bin = np.take(Y,indices,0)
        S_bin = np.take(S,indices,0)
        P_bin = np.take(P,indices,0)
        subset = [X_bin, Y_bin, S_bin, P_bin]

        assert np.count_nonzero(Y_bin==0) == 0

        return subset
    trainset = pick_binary_helper(trainset)
    valset = pick_binary_helper(valset)
    testset = pick_binary_helper(testset)
    return trainset, valset, testset


def generate_ts_fresh_features(trainset, valset, testset):
    # CAUTION: THIS FUNCTION WAS NOT TESTED!
    # trainset: training data
    # valset: validation data
    # testset: test data
    import numpy as np
    import pandas as pd
    from tsfresh import extract_features, select_features, extract_relevant_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
    import tsfresh.feature_extraction

    print("CAUTION: USING TS FRESH FEATURES WAS NOT TESTED AND IS THEREFORE NOT RECOMMENDED")
    user_input = input("If you want to continue press Y, if you want to stop press any other key")

    if not(user_input.lower() == 'y'):
        raise NotImplementedError

    X_train = trainset[0]
    X_val = valset[0]
    X_test = testset[0]
    Y_train = trainset[1]

    def make_ts_fresh_dataframe(X):
        m,n,r = X.shape
        X_arr = X.reshape((m*r,-1))
        timepoints = np.tile(np.arange(0,256, dtype=int),m)
        cut_trial = np.repeat(np.arange(0,m, dtype=int),r)
        X_arr = np.column_stack((cut_trial, timepoints, X_arr))
        X_df = pd.DataFrame(X_arr, columns=['id','time', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'])
        X_df['id'] = X_df['id'].astype(int)
        X_df['time'] = X_df['time'].astype(int)
        return X_df

    def make_ts_fresh_series(Y):
        return pd.Series(Y)

    settings = EfficientFCParameters()
    
    X_train_df = make_ts_fresh_dataframe(X_train)
    X_val_df = make_ts_fresh_dataframe(X_val)
    X_test_df = make_ts_fresh_dataframe(X_test)
    Y_train_series = make_ts_fresh_series(Y_train)
    X_train_features = extract_features(
                            X_train_df,
                            column_id="id",
                            column_sort="time",
                            n_jobs=4,
                            chunksize=2,
                            default_fc_parameters=settings
                            )
    impute(X_train_features)
    X_train_selected_features = select_features(X_train_features, Y_train_series)
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(X_train_selected_features)
    X_val_features = extract_features(
                            X_val_df,
                            column_id="id",
                            column_sort="time",
                            n_jobs=4,
                            chunksize=2,
                            kind_to_fc_parameters=kind_to_fc_parameters
                            )
    X_test_features = extract_features(
                            X_test_df,
                            column_id="id",
                            column_sort="time",
                            n_jobs=4,
                            chunksize=2,
                            kind_to_fc_parameters=kind_to_fc_parameters
                            )
    
    X_train_selected_features = X_train_selected_features.reindex(sorted(X_train_selected_features.columns), axis=1)
    X_val_features = X_val_features.reindex(sorted(X_val_features.columns), axis=1)
    X_test_features = X_test_features.reindex(sorted(X_test_features.columns), axis=1)
    
    assert np.all(X_train_selected_features.columns==X_val_features.columns)
    assert np.all(X_train_selected_features.columns==X_test_features.columns)
    assert np.all(X_val_features.columns==X_test_features.columns)

    trainset = X_train_selected_features.to_numpy(), trainset[1], trainset[2], trainset[3]
    valset = X_val_features.to_numpy(), valset[1], valset[2], valset[3]
    testset = X_test_features.to_numpy(), testset[1], testset[2], testset[3]

    print(X_train_selected_features.shape)
    print(X_val_features.shape)
    print(X_test_features.shape)
    
    return trainset, valset, testset
#!/usr/bin/env python
import tensorflow as tf
import tensorflow
import keras
import keras.backend as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
from tqdm import tqdm, trange
from helper_tb_logs import get_confusion_matrix

keras.backend.set_image_data_format('channels_last')


class AdversarialCNN:
    def __init__(self, chans, samples, n_output, n_nuisance, architecture='DeepConvNet', adversarial=False, lam=0.01):
        ### Parameters ###
        # chans: Number of EEG chanels
        # samples: Sample points of EEG Time Series
        # n_output: Number of Label Classes of the prediction, e.g. number different Emotions
        # n_nuisance: Number of Label Classes of the nuisance, e.g. number of used data-sources
        # architecture: Encoder and corresponding classifier Architecture
        # adversarial: wheter or not to use an adversarial network
        # lam: weighting parameter for the adversarial loss

        if lam == 0:
            print("Adversarial Training was set to False, however an Adversary is going to be trained for monitoring")
            adversarial = False
            lam = 1.0


        # Input, data set and model training scheme parameters
        self.chans = chans
        self.samples = samples
        self.n_output = n_output
        self.n_nuisance = n_nuisance
        self.lam = lam
        self.architecture = architecture

        # Build the network blocks
        self.enc = self.encoder_model(architecture)
        self.latent_dim = self.enc.output_shape[-1]  # inherit latent dimensionality
        self.cla = self.classifier_model(architecture)
        self.adv = self.adversary_model()

        # Compile the network with or without adversarial censoring
        input = Input(shape=(self.chans, self.samples, 1))
        latent = self.enc(input)
        output = self.cla(latent)
        leakage = self.adv(latent)
        self.adv.trainable = False
        self.acnn = Model(input, [output, leakage])

        if adversarial:
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True)],
                              loss_weights=[1., -1. * self.lam],
                              optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])
        else:   # trains a regular (non-adversarial) CNN, but will monitor leakage via the adversary alongside
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True)],
                              loss_weights=[1., 0.],
                              optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])

        self.adv.trainable = True
        self.adv.compile(loss=lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                         loss_weights=[self.lam],
                         optimizer=Adam(lr=1e-3, decay=1e-4),
                         metrics=['accuracy'])

    def encoder_model(self, architecture):
        model = Sequential()
        if architecture == 'EEGNet':
            model.add(Conv2D(8, (1, 32), padding='same', use_bias=False))
            model.add(BatchNormalization(axis=3))
            model.add(DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.)))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 4)))
            model.add(Dropout(0.25))
            model.add(SeparableConv2D(16, (1, 16), use_bias=False, padding='same'))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 8)))
            model.add(Dropout(0.25))
            model.add(Flatten())
        elif architecture == 'DeepConvNet':
            model.add(Conv2D(25, (1, 5) ) )
            model.add(Conv2D(25, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(50, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(100, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(200, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
        elif architecture == 'ShallowConvNet':
            model.add(Conv2D(40, (1, 13)))
            model.add(Conv2D(40, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation(lambda x: tf.square(x)))
            model.add(AveragePooling2D(pool_size=(1, 35), strides=(1, 7)))
            model.add(Activation(lambda x: tf.log(tf.clip(x, min_value=1e-7, max_value=10000))))
            model.add(Dropout(0.5))
            model.add(Flatten())
        elif architecture == 'MLP':
            model.add(Flatten())
            model.add(Dense(2184, kernel_initializer='he_normal'))
            model.add(GaussianNoise(0.005))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1310, kernel_initializer='he_normal'))
            model.add(GaussianNoise(0.005))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(786, kernel_initializer='he_normal'))
            model.add(GaussianNoise(0.005))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(472, kernel_initializer='he_normal'))
            model.add(GaussianNoise(0.005))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
        else:
            try:
                raise NotImplementedError
            finally:
                print("Unknown Architecture used")

        input = Input(shape=(self.chans, self.samples, 1))
        latent = model(input)

        return Model(input, latent, name='enc')

    def classifier_model(self, architecture):
        latent = Input(shape=(self.latent_dim,))
        if architecture == 'EEGNet':
            output = Dense(self.n_output, kernel_constraint=max_norm(0.25))(latent)
        elif architecture == 'DeepConvNet' or architecture == 'ShallowConvNet' or architecture == 'MLP':
            output = Dense(self.n_output)(latent)
        else:
            try:
                raise NotImplementedError
            finally:
                print("Unknown Architecture used")

        return Model(latent, output, name='cla')

    def adversary_model(self):
        latent = Input(shape=(self.latent_dim,))
        leakage = Dense(self.n_nuisance)(latent)

        return Model(latent, leakage, name='adv')

    def fit(self, train_set, val_set, logdir, epochs= 500, batch_size = 10, callbacks = None):
        x_train, y_train, p_train = train_set #last two components are adversary
        x_test, y_test, p_test = val_set

        self.acnn.fit([x_train], [y_train, p_train], validation_data= ([x_test], [y_test, p_test]), 
                        epochs = epochs, batch_size= batch_size, callbacks=callbacks, verbose = 2)


    def train(self, train_set, val_set, log, early_stopping_after_epochs=10, epochs=500, batch_size=40, run_name=''):
        self.writer = tensorflow.summary.create_file_writer(log+'/../tensorboard_logs/'+run_name+'/')

        monitoring_nb_val_acc = list()
        monitoring_nn_val_acc = list()

        nn_clf = KNeighborsClassifier(n_neighbors=5)
        nb_clf = GaussianNB()

        x_train, y_train, s_train = train_set
        x_test, y_test, s_test = val_set

        train_index = np.arange(y_train.shape[0])
        train_batches = [(i * batch_size, min(y_train.shape[0], (i + 1) * batch_size))
                         for i in range((y_train.shape[0] + batch_size - 1) // batch_size)]
        
        # Early stopping variables
        es_wait = 0
        es_best = np.Inf
        es_best_weights = None


        for epoch in range(1, epochs + 1):
            print('[{} - {} - {}] Epoch {}/{}'.format(run_name, self.architecture, str(self.lam), epoch, epochs))
            np.random.shuffle(train_index)
            train_log = []
            for iter, (batch_start, batch_end) in enumerate(tqdm(train_batches)):
                batch_ids = train_index[batch_start:batch_end]
                x_train_batch = x_train[batch_ids]
                y_train_batch = y_train[batch_ids]
                s_train_batch = s_train[batch_ids]
                z_train_batch = self.enc.predict_on_batch(x_train_batch)
                self.adv.train_on_batch(z_train_batch, s_train_batch)
                train_log.append(self.acnn.train_on_batch(x_train_batch, [y_train_batch, s_train_batch]))

            train_log = np.mean(train_log, axis=0)
            val_log = self.acnn.test_on_batch(x_test, [y_test, s_test])

            [y_pred_train, s_pred_train] = self.acnn.predict(x_train)
            [y_pred_val, s_pred_val] = self.acnn.predict(x_test)

            monitoring = True
            if monitoring:
                z_train = self.enc.predict(x_train)
                z_test = self.enc.predict(x_test)
                nn_clf.fit(z_train, np.argmax(s_train, axis=1))
                nb_clf.fit(z_train, np.argmax(s_train, axis=1))
                monitoring_nb_val_acc.append(nb_clf.score(z_test, np.argmax(s_test, axis=1)))
                monitoring_nn_val_acc.append(nn_clf.score(z_test, np.argmax(s_test, axis=1)))

            # Logging model training information per epoch
            print("[%s - %s - %s] Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                  % (run_name, self.architecture, str(self.lam), train_log[0], train_log[1], 100*train_log[3], train_log[2], 100*train_log[4]))
            print("[%s - %s - %s] Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                  % (run_name, self.architecture, str(self.lam), val_log[0], val_log[1], 100*val_log[3], val_log[2], 100*val_log[4]))
            with open(log + '/train.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(train_log[0]) + ',' + str(train_log[1]) + ',' +
                        str(100*train_log[3]) + ',' + str(train_log[2]) + ',' + str(100*train_log[4]) + '\n')
            with open(log + '/validation.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(val_log[0]) + ',' + str(val_log[1]) + ',' +
                        str(100*val_log[3]) + ',' + str(val_log[2]) + ',' + str(100*val_log[4]) + '\n')


            # Logging data to tensorboard
            with self.writer.as_default():
                with tf.name_scope("Encoder"):
                    for layer in self.acnn.layers[1].layers[1].layers:
                        if not(len(layer.trainable_weights)==0):
                            tensorflow.summary.histogram("Encoder %s /weights"%layer.name, layer.get_weights()[0], step=epoch)
                with tf.name_scope("Classifier"):
                    tensorflow.summary.histogram("Classifier layer-%s /weights"%self.acnn.layers[2].layers[1].name, self.acnn.layers[2].layers[1].get_weights()[0], step=epoch)
                with tf.name_scope("ACNN Adversary"):
                    tensorflow.summary.histogram("Adversary layer-%s /weights"%self.acnn.layers[3].layers[1].name, self.acnn.layers[3].layers[1].get_weights()[0], step=epoch)
                with tf.name_scope("ADV Adversary"):
                    tensorflow.summary.histogram("Adversary layer-%s /weights"%self.adv.layers[1].name, self.adv.layers[1].get_weights()[0], step=epoch)

                with tf.name_scope('Losses'):
                    tensorflow.summary.scalar("ACNN Train-Loss",train_log[0], step=epoch)
                    tensorflow.summary.scalar("CLA Train-Loss",train_log[1], step=epoch)
                    tensorflow.summary.scalar("ADV Train-Loss",train_log[2], step=epoch)
                    tensorflow.summary.scalar("ACNN Validation-Loss",val_log[0], step=epoch)
                    tensorflow.summary.scalar("CLA Validation-Loss",val_log[1], step=epoch)
                    tensorflow.summary.scalar("ADV Validation-Loss",val_log[2], step=epoch)
                with tf.name_scope('Accuracies'):
                    tensorflow.summary.scalar("Classifier-Accuracy (Train)",train_log[3], step=epoch)
                    tensorflow.summary.scalar("Adversary-Accuracy (Train)",train_log[4], step=epoch)
                    tensorflow.summary.scalar("Classifier-Accuracy (Val)",val_log[3], step=epoch)
                    tensorflow.summary.scalar("Adversary-Accuracy (Val)",val_log[4], step=epoch)

                cm_cla_train = get_confusion_matrix(y_pred_train, y_train)
                cm_adv_train = get_confusion_matrix(s_pred_train, s_train)
                cm_cla_val = get_confusion_matrix(y_pred_val, y_test)
                cm_adv_val = get_confusion_matrix(s_pred_val, s_test)
                with tf.name_scope("Classifier Train - Confusion Matrices"):
                    tensorflow.summary.image("Classifier Train", cm_cla_train, step=epoch)
                with tf.name_scope("Classifier Validation - Confusion Matrices"):
                    tensorflow.summary.image("Classifier Validation", cm_cla_val, step=epoch)
                with tf.name_scope("Adversary Train - Confusion Matrices"):
                    tensorflow.summary.image("Adversary Train", cm_adv_train, step=epoch)
                with tf.name_scope("Adversary Validation - Confusion Matrices"):
                    tensorflow.summary.image("Adversary Validation", cm_adv_val, step=epoch)

            self.writer.flush()

            # Check early stopping criteria based on validation CLA loss - patience for 10 epochs
            if np.less(val_log[1], es_best):
                es_wait = 0
                es_best = val_log[1]
                es_best_weights = self.acnn.get_weights()
            else:
                es_wait += 1
                if es_wait >= early_stopping_after_epochs:
                    print('Early stopping...')
                    self.acnn.set_weights(es_best_weights)
                    break
            print("Early Stopping in %s Epochs" % (early_stopping_after_epochs-es_wait))
        
        if monitoring:
            monitoring = pd.DataFrame()
            monitoring['nb Acc'] = monitoring_nb_val_acc
            monitoring['nn Acc'] = monitoring_nn_val_acc
            monitoring.to_csv(os.path.join(log, 'leakage.csv'))
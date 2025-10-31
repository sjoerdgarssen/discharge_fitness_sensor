import os
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()
import numpy as np
import random
import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPool1D, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import initializers
from tensorflow.keras.metrics import AUC

class FeatureSetSelection(object):
    '''
    Selects features from a dataset based on predefined feature sets, removing features without variance.
    Provides methods to fit the selection criteria to the data and transform the data to include only the selected features.
    '''
    def __init__(self, feature_sets=['Set1', 'Set2']):
        self.feature_sets = feature_sets

    def fit(self, X, y=None, sample_weight=None):

        # Store input feature names
        self.features_in = list(X.columns)

        # Remove features without variance
        varthres = VarianceThreshold(threshold=0).fit(X)
        X = X.loc[:, varthres.get_feature_names_out()]

        # Add sensor features if present
        feature_sets = self.feature_sets + ['Set5', 'nn_feature']
        self.columns =[col for col in X.columns if any(sub in col for sub in feature_sets)]

        return self

    def transform(self, X):

        # Return the DataFrame with only the selected features
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):

        # Return the names of the selected features
        return np.asarray(self.columns)

class CorrelationFeatureSelection(object):
    '''
    Removes features that are highly correlated with each other.
    Keeps the features with the highest absolute correlation with the outcome and removes others.
    '''

    def __init__(self, n_features=20, indices=True, correlation_threshold=0.7, sample_weight=None,
                 feature_sets=['Set1', ' Set2']):
        self.n_features = n_features
        self.indices = indices
        self.cor_thres = correlation_threshold
        self.sample_weight = sample_weight
        self.feature_sets = feature_sets

    def get_weights(self, Xy=None, sample_weight=None):
        '''
        function to get class weights and multiply it by sample weights, such that each patient within a class is
        equally important, and each class is equally important in calculating Pearson coefficients
        :param Xy: dataframe including 'y' column
        :param sample_weight: sample weights
        :return: weights
        '''
        class_weights = compute_sample_weight(class_weight='balanced', y=Xy['y'].values)
        weights = class_weights * sample_weight
        return weights

    def weighted_mean(self, x=None, w=None):
        '''
        calculate a weighted mean
        :param x: array
        :param w: weight array
        :return: weighted mean
        '''
        return np.sum(x * w) / np.sum(w)

    def weighted_cov(self, x=None, y=None, w=None):
        '''
        calculate weighted covariance between x and y
        :param x: array x
        :param y: array y
        :param w: weights
        :return: weighted covariance
        '''
        return np.sum(w * (x - self.weighted_mean(x, w)) * (y - self.weighted_mean(y, w))) / np.sum(w)

    def weighted_corr(self, x=None, y=None, w=None):
        '''
        calculates weighted correlation coefficient for x and y
        :param x: array x
        :param y: array y
        :param w: weights
        :return: weighted correlation coefficient
        '''
        return self.weighted_cov(x, y, w) / np.sqrt(self.weighted_cov(x, x, w) * self.weighted_cov(y, y, w))

    def get_feature_names_out(self):
        '''
        :return: selected features
        '''
        return self.cols

    def fit(self, X, y, sample_weight):
        '''
        Selects features based on weighted correlation coefficients with outcome and inter-feature correlation. Also
        scales features
        :param X: input dataframe
        :param y: outcome array
        :param sample_weight: balanced array with sample weights
        :return: updates self with the selected features
        '''

        # Store input feature names
        self.features_in = list(X.columns)

        # Remove features without variance
        preselection = FeatureSetSelection(feature_sets=self.feature_sets).fit(X)
        X = X.loc[:, preselection.get_feature_names_out()]

        # Calculate absolute weighted correlation coefficients
        Xy = X.copy()
        Xy.loc[:, 'y'] = y
        corr = pd.DataFrame(columns=X.columns, index=X.columns)
        weights = self.get_weights(Xy=Xy, sample_weight=sample_weight)
        for ix1, v1 in enumerate(Xy.columns):
            corr.loc[v1, v1] = float(1)
            for ix2, v2 in enumerate(Xy.columns[ix1+1:]):
                corcoef = self.weighted_corr(x=Xy[v1].values, y=Xy[v2].values, w=weights)
                corr.loc[v1, v2] = abs(corcoef)
                corr.loc[v2, v1] = abs(corcoef)

        # Select features with highest correlation with outcome. If feature is correlated with a higher ranked feature,
        # it is ignored. Higher rank means higher absolute Pearson coefficient with outcome.
        corr.sort_values('y', inplace=True, ascending=False)
        corr = corr.loc[corr.index.values, corr.index.values]
        corr.drop(index='y', columns='y', inplace=True)
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0] - 1):
            if columns[i]:
                for j in range(i + 1, corr.shape[0]):
                    if columns[j]:
                        if corr.iloc[i, j] >= self.cor_thres:
                            columns[j] = False

        # Store selected features
        self.cols = list(corr.columns[columns][:self.n_features])

        # Scale features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = X.loc[:, self.cols]
        self.scaler = scaler.fit(X=X)

        return self

    def transform(self, X):
        '''
        selects the right features for X
        :param X: input dataframe
        :return: dataframe with scaled selected features
        '''

        # select features
        X = X.loc[:, self.cols]

        # scale features
        X = self.scaler.transform(X=X)

        # return dataframe
        X = pd.DataFrame(data=X, columns=self.cols)

        return X


class CNN_Model(object):
    '''
    Builds and trains a CNN model. Also used for extracting features from the trained CNN model.
    '''

    def __init__(self, nr_features, seed_nr):
        self.nr_features = nr_features
        self.seed_nr = seed_nr

    def fit_scaling(self, X):
        '''
        functions that calculates the min and max for each feature to be able to 0-1 scale
        :param X: training sequences
        :return:
        '''
        self.min_values = np.min(X, axis=(0, 1), keepdims=True)
        self.max_values = np.max(X, axis=(0, 1), keepdims=True)

    def scale(self, X):
        '''
        scales X to 0-1
        :param X: sequences
        :return:
        '''
        X = (X - self.min_values) / (self.max_values - self.min_values)
        return X

    def compute_weights(self, labels, subject_ids):
        '''
        Compute class weights and sample weights and merge them by multiplication
        :param labels: sequence with labels
        :param subject_ids: sequence of patient ids
        return np.array of weights for each patient
        '''

        # Ensure labels and subject_ids are 1D arrays
        labels = np.asarray(labels).flatten()
        subject_ids = np.asarray(subject_ids).flatten()

        # Compute class weights
        class_weights = {label: len(labels) / (2.0 * count) for label, count in Counter(labels).items()}
        class_weights = np.array([class_weights[label] for label in labels.flatten()])

        # Compute patient weights
        subject_counts = Counter(subject_ids)
        subject_weights = {subject: 1 / count for subject, count in subject_counts.items()}

        # Assign sample weights based on subject
        subject_weights = np.array([subject_weights[subject] for subject in subject_ids])
        weight_average = np.mean(subject_weights)
        subject_weights = subject_weights / weight_average

        # merge class and sample weights
        subject_weights *= class_weights

        return subject_weights

    def build_cnn(self):
        '''
        builds the CNN
        '''

        # architecture
        input = Input(batch_shape=(None, 32, self.nr_features), name='Input')
        cnn = Conv1D(filters=32, kernel_size=8, strides=1, padding='causal', activation=ReLU(),
                     name='CNN', kernel_initializer=initializers.HeNormal(seed=self.seed_nr), dilation_rate=1,
                     kernel_regularizer=l2(0.01), data_format='channels_last')(input)
        print("CNN Output Shape:", cnn.shape)
        pooling = GlobalMaxPool1D(name='extraction_layer')(cnn)
        print("Pooling Output Shape:", pooling.shape)
        drop_out_final = Dropout(0.2, name='dropout_layer')(pooling)
        output = Dense(1, activation='sigmoid', name='Label', kernel_regularizer=l2(0.01),
                       kernel_initializer=initializers.GlorotNormal())(drop_out_final)

        # compile
        model = Model(inputs=input, outputs=output, name='CNN')
        model.summary()
        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

        # store
        self.model = model

    def train_nn(self, x, y, subjects, x_val, y_val):
        '''
        function that fits cnn model
        :param model: cnn model developed by build_cnn()
        :param x: train sequences
        :param y: train labels
        :param subjects: sequence of training subjects
        :param x_val: validation sequences
        :param y_val: validation labels
        '''

        # fit scaler and transform data
        self.fit_scaling(x)
        x = self.scale(x)
        x_val = self.scale(x_val)

        # use early stopping rule
        monitor = 'val_loss'
        early_stopping = EarlyStopping(
            monitor=monitor, mode='min', patience=10, verbose=0, restore_best_weights=True, min_delta = 0.001,
            start_from_epoch=20
        )

        # get weights
        sample_weights = self.compute_weights(labels=y, subject_ids=subjects)

        self.model.fit(x=x, y=y, batch_size=16, shuffle=True, validation_data=(x_val, y_val), epochs=100,
                            callbacks=[early_stopping], verbose=1, sample_weight=sample_weights)


    def develop_cnn(self, train_sequences, train_labels, train_subjects, val_sequences, val_labels):
        '''
        function that builds and fits a CNN model
        :param train_sequences: training sequences
        :param train_labels: training labels sequence
        :param: train_subjects: training subjects sequence
        :param val_sequences: validation sequences for Early Stopping rule
        :param val_labels: validation labels
        '''

        # Set random states using the random_state instance
        random.seed(self.seed_nr)
        np.random.seed(self.seed_nr)
        tf.random.set_seed(self.seed_nr)

        # build model
        self.build_cnn()

        # fit model
        self.train_nn(x=train_sequences, y=train_labels, subjects=train_subjects, x_val=val_sequences,
                             y_val=val_labels)

    def extract_nn_features(self, X, ts, sub):
        '''
        Extracts features from the CNN
        :param model: CNN model
        :param X: sequences
        :param ts: timestamp sequences of X
        :param sub: subject sequences of X
        :return: dataframe with subjects, timestamps and 32 neural features
        '''

        # make model that has the extraction layer as output
        feature_extraction_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('extraction_layer').output
        )

        # scale X
        X = self.scale(X)

        # extract features for all sequences in X
        features = feature_extraction_model.predict(X)

        # store in dataframe
        return_df = pd.DataFrame({'ts': ts[:, 0], 'subject': sub[:, 0]})
        feature_names = [f'nn_feature_{x}' for x in range(1, 33)]
        return_df[feature_names] = features

        return return_df

    def add_nn_features(self, X, ts, sub, df_features):
        '''
        function that adds neural features to dataframe with other features
        :param X: sequences to extract neural features from, should be of the same subjects as df_features
        :param ts: sequences of Timestamps
        :param sub: sequences of subject IDs
        :param df_features: dataframe with the other features of the same subjects. Should be index by ts, subject and
        session
        :return: df_features including neural features
        '''

        # extract neural features
        nn_features = self.extract_nn_features(X, ts, sub)

        # combine neural features with other features        # df_features.reset_index(inplace=True)
        df_features['ts'] = df_features['ts'].values.astype('datetime64[ns]')
        df_features = df_features.merge(nn_features, on=['ts', 'subject'], how='left')
        df_features = df_features.groupby('subject').apply(lambda group: group.ffill()).reset_index(drop=True)
        df_features.set_index(['ts', 'subject'], inplace=True)

        return df_features

def get_pipelines(random_state=None, feature_sets=True):
    '''
    Generate pipelines for Random Forest, Logistic Regression, and XGBoost with different configurations.
    :param random_state: RandomState instance for reproducibility.
    :param feature_sets: Boolean to indicate whether EMR feature sets are optimized.
    :return: Dictionary of pipelines.
    '''

    # Initialize dictionary to store pipelines
    pipelines = {}

    # Initialize feature sets
    if feature_sets: # if optimized
        feature_sets = [['Set1', 'Set2'], ['Set1', 'Set2', 'Set3'], ['Set1', 'Set2', 'Set4'], ['Set1', 'Set2', 'Set3', 'Set4']]
    else: # if not optimized (for Sensor models)
        feature_sets = [['Set1', 'Set2', 'Set3', 'Set4']]

    # Random Forest pipelines
    i = 0
    for depth in [5, 10]:
        for feature_set in feature_sets:

            # Construct the key
            key = f'RF-{i}'

            # Create the pipeline for the given parameters
            pipeline = Pipeline([
                ('preselection', FeatureSetSelection(feature_sets=feature_set)),#VarianceThreshold(threshold=0)),
                ('classification', RandomForestClassifier(random_state=random_state, class_weight='balanced',
                                                          max_depth=depth, n_estimators=500))
            ])

            # Add the pipeline to the dictionary
            pipelines[key] = pipeline

            i += 1

    # Logistic Regression pipelines
    i = 0
    for feature_set in feature_sets:
        for n_features in [10, 20]:

            key = f'LR-{i}'
            # Case when no regularization (penalty=None)
            pipeline = Pipeline([
                ('feature_selection', CorrelationFeatureSelection(n_features=n_features, feature_sets=feature_set)),
                ('classification',
                 LogisticRegression(penalty=None, class_weight='balanced', random_state=random_state,
                                    max_iter=10000))
            ])
            pipelines[key] = pipeline
            i += 1

    # XGB
    i = 0
    for feature_set in feature_sets:
        for depth in [5, 10]:
            for reg_lambda in [1, 10]:
                for learning_rate in [0.1, 0.3]:
                    key = f'XGB-{i}'

                    # Create the pipeline for the given parameters
                    pipeline = Pipeline([
                        ('preselection', FeatureSetSelection(feature_sets=feature_set)),#VarianceThreshold(threshold=0)),
                        ('classification', XGBClassifier(
                            use_label_encoder=False,
                            eval_metric='logloss',
                            n_estimators=500,
                            min_child_weight=1,
                            random_state=random_state,
                            learning_rate=learning_rate,
                            max_depth=depth,
                            colsample_bytree=0.5,
                            reg_lambda=reg_lambda,
                            reg_alpha=0,
                            verbosity=0,
                            tree_method='hist'
                        ))
                    ])

                    # Add the pipeline to the dictionary
                    pipelines[key] = pipeline

                    # Increment model counter
                    i += 1

    return pipelines

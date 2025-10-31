import os
os.environ['PYTHONHASHSEED'] = '42'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import tensorflow as tf
import copy

# Set all seeds globally
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from utils.data_processing import *
from utils.train import train_optimized_model, train_models#, store_optimal_model_settings
from utils.pipelines import get_pipelines, CNN_Model#, get_optimized_pipelines
from dummy.utils.test import *

def train_test_split_patient_level(df, random_state, k):
    '''
    Splits patients into k folds with stratified class distribution.
    :param df: DataFrame with patient-level labels (one row per patient) and column `patient_y` indicating label.
    :param random_state: Integer or numpy RandomState instance for reproducibility.
    :param k: Number of folds for k-fold cross-validation.
    :return: DataFrame with an additional column indicating the fold assignment.
    '''
    # reset index and add empty fold column
    df.reset_index(inplace=True, drop=True)
    df['fold'] = -1

    # partition data over different folds with equal class ratio
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(kf.split(df, df['patient_y'])):
        df.loc[df.index.isin(val_idx), 'fold'] = fold

    return df

def nestedCV(df_sensor=pd.DataFrame(), df_features=pd.DataFrame(), repeats=5, k=5, random_state=None):
    '''
    Applies nested cross-validation in a repeated way.
    :param df_sensor: DataFrame with resampled sensor data of all patients, with at least `subject` and `ts` as index.
    :param df_features: DataFrame with all other features of all patients, with at least `subject` and `ts` as index.
    :param repeats: Number of repeats for cross-validation.
    :param k: Number of folds for k-fold cross-validation.
    :param random_state: Integer or numpy RandomState instance for reproducibility.
    :return: Metrics DataFrame, predictions, feature importances, and model optimization settings.
    '''

    # Extract sequences for all patients and filter valid ones
    _, _, subjects, _ = get_sequences_nn(df=df_sensor, train=False)

    # Get patient-level labels and filter on non-neutral patients and valid sequences
    patient_labels = get_patient_label(df=df_features)
    patient_labels = patient_labels.loc[(patient_labels['patient_y'] != 0) &
                                        (patient_labels['subject'].isin(subjects[:,0])), :]

    # Initialize storage for metrics, predictions, feature importances, and model optimization
    metrics_outerloop = pd.DataFrame()
    all_predictions = {feature_set: {algorithm: pd.DataFrame() for algorithm in ['LR', 'RF', 'XGB']} for
                       feature_set in ['emr', 'emr_sensor', 'sensor']}
    feature_importances = copy.deepcopy(all_predictions)
    model_optimization_df = pd.DataFrame(columns=['feature_set', 'algorithm', 'modelname', 'fold', 'repeat'])

    # Get optimization pipelines for different feature sets
    optimization_pipelines = get_pipelines(random_state=random_state)
    optimization_pipelines_sensor = get_pipelines(random_state=random_state, feature_sets=False)

    # Iterate over repeats
    for repeat in range(repeats):

        # Split patients into k folds
        patient_labels = train_test_split_patient_level(df=patient_labels, random_state=random_state, k=k)

        # Iterate over outer CV loops
        for outer_test in range(k):

            # Initialize storage for optimization results
            optimization_emr_sensor = {algorithm: {'auc': 0, 'cnn': None, 'pipeline': None, 'modelname': None} for
                                       algorithm in ['LR', 'RF', 'XGB']}
            optimization_sensor = copy.deepcopy(optimization_emr_sensor)
            optimization_emr = {k: [] for k in optimization_pipelines.keys()}

            # Inner loop: Use remaining folds for training and validation
            patient_labels_inner_cv = patient_labels.loc[patient_labels['fold'] != outer_test, :]

            # iterate over inner CV loops
            for inner_val in patient_labels_inner_cv['fold'].unique():

                # Split sujects into training and validation subjects
                train_subjects = patient_labels_inner_cv.loc[patient_labels_inner_cv['fold'] != inner_val, 'subject'].unique()
                val_subjects = patient_labels_inner_cv.loc[patient_labels_inner_cv['fold'] == inner_val, 'subject'].unique()

                # Filter feature data for training and validation
                df_features_train = df_features.loc[df_features['subject'].isin(train_subjects), :]
                df_features_val = df_features.loc[df_features['subject'].isin(val_subjects), :]

                ### EMR+SENSOR DATA ###
                # Extract sequences for CNN training
                sequences_train, labels_train, subjects_train, timestamps_train = get_sequences_nn(df=df_sensor.loc[df_sensor['subject'].isin(train_subjects)], train=True)
                sequences_val, labels_val, subjects_val, timestamps_val = get_sequences_nn(df=df_sensor.loc[df_sensor['subject'].isin(val_subjects)], train=True)

                # Train CNN model
                random_int = random_state.randint(0,1000)
                CNN = CNN_Model(nr_features=sequences_train.shape[-1], seed_nr=random_int)
                CNN.develop_cnn(train_sequences=sequences_train, train_labels=labels_train,
                                train_subjects=subjects_train, val_sequences=sequences_val, val_labels=labels_val)

                # Add neural features to training and validation sets
                df_features_train = CNN.add_nn_features(sequences_train, timestamps_train, subjects_train,
                                                    df_features_train)
                df_features_val = CNN.add_nn_features(sequences_val, timestamps_val, subjects_val,
                                                  df_features_val)

                # Train models for EMR+Sensor data
                models = train_models(df=df_features_train, pipelines=optimization_pipelines, feature_set='emr_sensor',
                                      random_state=random_int)

                # Test models on validation set
                model_tests = test_models(df=df_features_val, models=models, outerloop=False, feature_set='emr_sensor')

                # Assess performance at the patient level
                auc_storage = assess_patient_level_performance(test_predictions=model_tests,
                                                               df_ground_truth=patient_labels_inner_cv)

                # Update optimization results if performance improves
                optimization_emr_sensor = update_optimization(auc_storage=auc_storage, model_cnn=CNN,
                                                              optimization=optimization_emr_sensor,
                                                              pipelines=optimization_pipelines)

                ### SENSOR DATA ###
                # Train models for Sensor data
                models = train_models(df=df_features_train, pipelines=optimization_pipelines_sensor,
                                      feature_set='sensor', random_state=random_int)

                # Test models on validation set
                model_tests = test_models(df=df_features_val, models=models, outerloop=False, feature_set='sensor')

                # Assess performance at the patient level
                auc_storage = assess_patient_level_performance(test_predictions=model_tests,
                                                               df_ground_truth=patient_labels_inner_cv)

                # Update optimization results if performance improves
                optimization_sensor = update_optimization(auc_storage=auc_storage, model_cnn=CNN,
                                                              optimization=optimization_sensor,
                                                              pipelines=optimization_pipelines_sensor)

                ### EMR DATA ###
                # Train models for EMR data
                models = train_models(df=df_features_train, pipelines=optimization_pipelines, feature_set='emr',
                                      random_state=random_int)

                # Test models on validation set
                model_tests = test_models(df=df_features_val, models=models, outerloop=False, feature_set='emr')

                # Assess performance at the patient level
                auc_storage = assess_patient_level_performance(test_predictions=model_tests,
                                                               df_ground_truth=patient_labels_inner_cv)

                # Store performance metrics
                for key, value in optimization_emr.items():
                    value.append(auc_storage[key])

                # Delete variables to free up memory
                del auc_storage, CNN, model_tests

                # end of inner CV loop

            # Continue the outer cross-validation loop
            random_int = random_state.randint(0, 1000)

            # Get the subjects for training and testing based on the current outer test fold
            train_subjects = patient_labels.loc[patient_labels['fold'] != outer_test, 'subject'].unique()
            test_subjects = patient_labels.loc[patient_labels['fold'] == outer_test, 'subject'].unique()

            # Extract sequences for training and testing
            # Training sequences are extracted from the sensor data for the training subjects
            sequences_train, labels_train, subjects_train, timestamps_train = get_sequences_nn(
                df=df_sensor.loc[df_sensor['subject'].isin(train_subjects)], train=True)

            # Testing sequences are extracted from the sensor data for the testing subjects
            sequences_test, labels_test, subjects_test, timestamps_test = get_sequences_nn(
                df=df_sensor.loc[df_sensor['subject'].isin(test_subjects)], train=False)

            # Filter the feature DataFrame to include only the training and testing subjects
            df_features_train = df_features.loc[df_features['subject'].isin(train_subjects), :]
            df_features_test = df_features.loc[df_features['subject'].isin(test_subjects), :]

            ### EMR&SENSOR OPTIMIZATION AND OUTER CV PREDICTION ###
            # Iterate over the algorithms (Logistic Regression, Random Forest, XGBoost) and their corresponding optimizations
            for algorithm, optimizations in optimization_emr_sensor.items():

                # Add the model optimization results to the DataFrame. This includes the feature set, algorithm, model
                # name, repeat number, and fold number
                model_optimization_df = pd.concat([model_optimization_df, pd.DataFrame({'feature_set': 'emr_sensor',
                                                  'algorithm': algorithm, 'modelname': optimizations['modelname'],
                                                  'repeat': repeat, 'fold': outer_test}, index=[0])], ignore_index=True)

                # Add neural network features to the training and testing datasets. These features are generated by the
                # CNN model stored in the optimizations
                df_features_train_ = optimizations['cnn'].add_nn_features(sequences_train, timestamps_train, subjects_train,
                                                        df_features_train)
                df_features_test_ = optimizations['cnn'].add_nn_features(sequences_test, timestamps_test, subjects_test,
                                                      df_features_test)

                # Train the model using the optimized pipeline and update the feature importance
                # The trained model and updated feature importance are returned
                model, feature_importances['emr_sensor'][algorithm] = train_optimized_model(df=df_features_train_,
                        pipeline=optimizations['pipeline'], algorithm=algorithm, repeat=repeat, fold=outer_test,
                        feature_importance=feature_importances['emr_sensor'][algorithm], feature_set='emr_sensor',
                                                                                            random_state=random_int)

                # Test the trained model on the test dataset and store the predictions. The predictions are added to the
                # all_predictions dictionary
                model_tests, all_predictions['emr_sensor'] = test_models(df=df_features_test_, models={algorithm: model},
                                                        outerloop=True, all_predictions=all_predictions['emr_sensor'],
                                                        repeat=repeat, fold=outer_test, feature_set='emr_sensor')

                # Assess the model's performance at the patient level. This calculates metrics such as AUC, F1-score,
                # sensitivity, precision, and threshold
                metrics = assess_patient_level_performance(test_predictions=model_tests, df_ground_truth=patient_labels,
                                                        outerloop=True)

                # Store the calculated metrics in the `metrics_outerloop` DataFrame
                metrics_outerloop = pd.concat(
                    [metrics_outerloop, pd.DataFrame({'algorithm': algorithm, 'feature_set': 'emr_sensor',
                                                  'auc': metrics[algorithm]['auc'], 'f1': metrics[algorithm]['f1'],
                                                  'sensitivity': metrics[algorithm]['sensitivity'],
                                                  'precision': metrics[algorithm]['precision'],
                                                  'threshold': metrics[algorithm]['threshold'],
                                                  'repeat': repeat, 'fold': outer_test},
                                                 index=[0])], ignore_index=True)

            # Delete variables to free up memory
            del optimization_emr_sensor, model, model_tests

            ### SENSOR OPTIMIZATION AND OUTER CV PREDICTION ###
            # Iterate over the algorithms (Logistic Regression, Random Forest, XGBoost) and their corresponding optimizations
            for algorithm, optimizations in optimization_sensor.items():

                # Add the model optimization results to the DataFrame. This includes the feature set, algorithm, model
                # name, repeat number, and fold number
                model_optimization_df = pd.concat([model_optimization_df,
                                                   pd.DataFrame({'feature_set': 'sensor',
                                                                        'algorithm': algorithm,
                                                                        'modelname': optimizations[
                                                                            'modelname'],
                                                                        'repeat': repeat,
                                                                        'fold': outer_test},
                                                                       index=[0])], ignore_index=True)

                # Add neural network features to the training and testing datasets. These features are generated by the
                # CNN model stored in the optimizations
                df_features_train_ = optimizations['cnn'].add_nn_features(sequences_train, timestamps_train,
                                                                          subjects_train,
                                                                          df_features_train)
                df_features_test_ = optimizations['cnn'].add_nn_features(sequences_test, timestamps_test, subjects_test,
                                                                         df_features_test)

                # Train the model using the optimized pipeline and update the feature importance
                # The trained model and updated feature importance are returned
                model, feature_importances['sensor'][algorithm] = train_optimized_model(df=df_features_train_,
                                                                                            pipeline=optimizations[
                                                                                                'pipeline'],
                                                                                            algorithm=algorithm,
                                                                                            repeat=repeat,
                                                                                            fold=outer_test,
                                                                                            feature_importance=
                                                                                            feature_importances[
                                                                                                'sensor'][
                                                                                                algorithm],
                                                                                            feature_set='sensor',
                                                                                        random_state=random_int)

                # Test the trained model on the test dataset and store the predictions. The predictions are added to the
                # all_predictions dictionary
                model_tests, all_predictions['sensor'] = test_models(df=df_features_test_,
                                                                         models={algorithm: model},
                                                                         outerloop=True,
                                                                         all_predictions=all_predictions['sensor'],
                                                                         repeat=repeat, fold=outer_test,
                                                                         feature_set='sensor')

                # Assess the model's performance at the patient level. This calculates metrics such as AUC, F1-score,
                # sensitivity, precision, and threshold
                metrics = assess_patient_level_performance(test_predictions=model_tests, df_ground_truth=patient_labels,
                                                           outerloop=True)

                # Store the calculated metrics in the `metrics_outerloop` DataFrame
                metrics_outerloop = pd.concat(
                    [metrics_outerloop, pd.DataFrame({'algorithm': algorithm, 'feature_set': 'sensor',
                                                  'auc': metrics[algorithm]['auc'], 'f1': metrics[algorithm]['f1'],
                                                  'sensitivity': metrics[algorithm]['sensitivity'],
                                                  'precision': metrics[algorithm]['precision'],
                                                  'threshold': metrics[algorithm]['threshold'],
                                                  'repeat': repeat, 'fold': outer_test},
                                                 index=[0])],
                    ignore_index=True)

            # Delete variables to free up memory
            del optimization_sensor, model, model_tests

            ### EMR OPTIMIZATION AND OUTER CV PREDICTION ###
            # Store the optimal model settings for the EMR feature set
            # This function updates the `model_optimization_df` with the best pipelines for the current fold and repeat
            model_optimization_df, pipelines_emr_optimal = store_optimal_model_settings(df=model_optimization_df,
                optimization_emr=optimization_emr, optimization_pipelines=optimization_pipelines, fold=outer_test,
                repeat=repeat)

            # Train models using the optimal pipelines for the EMR feature set
            # The function returns the trained models and updates the feature importance for the EMR feature set
            models, feature_importances['emr'] = train_models(df=df_features_train_, pipelines=pipelines_emr_optimal,
                                                              outerloop=True,
                                  feature_importance=feature_importances['emr'], repeat=repeat, fold=outer_test,
                                                              feature_set='emr', random_state=random_int)

            # Test the trained models on the test dataset for the EMR feature set
            # The function returns the test predictions and updates the `all_predictions` dictionary
            model_tests, all_predictions['emr'] = test_models(df=df_features_test_, models=models,
                                                              outerloop=True, all_predictions=all_predictions['emr'],
                                                              repeat=repeat, fold=outer_test, feature_set='emr')

            # Assess the model's performance at the patient level for the EMR feature set
            # This function calculates metrics such as AUC, F1-score, sensitivity, precision, and threshold
            auc_storage = assess_patient_level_performance(test_predictions=model_tests, df_ground_truth=patient_labels,
                                                           outerloop=True)

            # Store the calculated metrics in the `metrics_outerloop` DataFrame
            # Each algorithm's metrics are added as a new row in the DataFrame
            for algorithm, metrics in auc_storage.items():
                metrics_outerloop = pd.concat(
                    [metrics_outerloop, pd.DataFrame({'algorithm': algorithm, 'feature_set': 'emr',
                                                  'auc': metrics['auc'], 'f1': metrics['f1'],
                                                  'sensitivity': metrics['sensitivity'],
                                                  'precision': metrics['precision'],
                                                  'threshold': metrics['threshold'],
                                                  'repeat': repeat, 'fold': outer_test},
                                                 index=[0])],
                    ignore_index=True)

            # Delete variables to free up memory
            del auc_storage, pipelines_emr_optimal

            # end of outer loop

        # end of repeat

    return metrics_outerloop, all_predictions, feature_importances, model_optimization_df

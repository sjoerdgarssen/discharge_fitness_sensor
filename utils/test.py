import copy
from dummy.utils.data_processing import *
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, classification_report


def test_models(df=pd.DataFrame(), models=None, all_predictions={}, outerloop=False, fold=0, repeat=0,
                feature_set='emr'):
    '''
    Function to predict the test set with all models stored in the `models` dictionary.
    :param df: DataFrame containing features (X) and labels (y).
    :param models: Dictionary with stored models. Keys must start with RF, LR, or XGB.
    :param all_predictions: Dictionary to store all predictions for each algorithm (used when `outerloop=True`).
    :param outerloop: Boolean indicating whether models are tested in the outer loop of nested CV.
    :param fold: Fold number (int).
    :param repeat: Repeat number (int).
    :param feature_set: Feature set type ('emr', 'emr_sensor', or 'sensor').
    :return: If `outerloop=True`, returns `test_predictions` and `all_predictions`. Otherwise, returns `test_predictions`.
    '''

    # Drop rows with missing values
    df = df.dropna(axis=0)

    # Remove sensor features if the feature set is 'emr'
    if feature_set == 'emr':
        df = df.drop(columns=df.filter(regex='HR|RR|Act|Pos_|nn_feature').columns)

    # Initialize a dictionary to store predictions for each model
    test_predictions = {key: None for key, value in models.items()}

    # Split the DataFrame into features (X) and labels (y)
    X, y = get_Xy(df)

    # Iterate over the models
    for key, model in models.items():

        # Select EMR feature sets based on the model type
        if 'RF' in key or 'XGB' in key:
            feature_sets = model.named_steps['preselection'].feature_sets
        else:
            feature_sets = model.named_steps['feature_selection'].feature_sets

        # Remove NaN and duplicates based on the EMR feature sets
        X_,  y_ = draw_df(X, y, feature_sets)

        # If the feature set is 'sensor', remove EMR features
        if feature_set == 'sensor':
            columns = [col for col in X_.columns if not any(sub in col for sub in ['Set1', 'Set2', 'Set3', 'Set4'])]
            X_ = X_[columns]

        # Predict probabilities and store in `test_predictions`
        y_pred = model.predict_proba(X_)[:, 1]
        test_predictions[key] = pd.DataFrame({'y_pred': y_pred, 'y': y_}, index=X_.index)

        # If testing in the outer loop, store additional data
        if outerloop:

            # Get feature importance and input data
            if 'RF' in key or 'XGB' in key:
                feature_names = model.named_steps['preselection'].get_feature_names_out()
                X_ = X_.loc[:, feature_names]
            elif 'LR' in key:
                feature_names = model.named_steps['feature_selection'].get_feature_names_out()
                X_ = X_.loc[:, feature_names]

            # Store input data
            if feature_set == 'emr_sensor' and feature_sets != ['Set1', 'Set2', 'Set3', 'Set4']:
                X_all, _ = draw_df(X, y, ['Set1', 'Set2', 'Set3', 'Set4'])
                df_ = pd.merge(X_all, test_predictions[key], left_index=True, right_index=True, how='left')
            else:
                df_ = pd.concat([test_predictions[key], X_], axis=1)

            # Add fold and repeat information
            df_['fold'] = fold
            df_['repeat'] = repeat
            all_predictions[key] = pd.concat([all_predictions[key], df_])

    if outerloop:
        return test_predictions, all_predictions
    else:
        return test_predictions

def assess_patient_level_performance(test_predictions, df_ground_truth, outerloop=False):
    '''
    Assess model performance at the patient level.
    :param test_predictions: Dictionary with model predictions. Keys are model names, values are DataFrames with predictions.
    :param df_ground_truth: DataFrame with patient-level ground truth labels.
    :param outerloop: Boolean indicating whether models are tested in the outer loop.
    :return: Dictionary with performance metrics.
    '''

    # Initialize a dictionary to store metrics for each model
    if outerloop:
        metric_storage = {key: {'auc': 0, 'sensitivity': 0, 'precision': 0, 'f1': 0, 'threshold': 0} for key, value in test_predictions.items()}
    else:
        metric_storage = {key: 0 for key, value in test_predictions.items()}

    # Iterate over each model
    for key, prediction in test_predictions.items():

        # Get the maximum prediction for each patient, since a patient is considered a true or false positive if at
        # least one prediction was positive
        y_pred_max = prediction.groupby('subject')['y_pred'].max().reset_index()

        # Merge predictions with ground truth labels
        df_ground_truth_ = df_ground_truth.merge(y_pred_max, on='subject', how='right')
        df_ground_truth_['patient_y'] = df_ground_truth_['patient_y'].astype(int)
        auc_ = roc_auc_score(y_true=df_ground_truth_['patient_y'], y_score=df_ground_truth_['y_pred'])

        if not outerloop:
            metric_storage[key] = auc_

        else:

            # Calculate the threshold for which specificity is at least 87%
            fpr, tpr, thresholds = roc_curve(y_true=df_ground_truth_['patient_y'], y_score=df_ground_truth_['y_pred'],
                                             drop_intermediate=False)
            specificity = 1 - fpr
            threshold_index = np.argmin(specificity >= 0.87) - 1
            metric_storage[key]['specificity'] = specificity[threshold_index]
            metric_storage[key]['sensitivity'] = tpr[threshold_index]
            metric_storage[key]['threshold'] = thresholds[threshold_index]
            metric_storage[key]['auc'] = auc_

    return metric_storage

def update_optimization(auc_storage, model_cnn, optimization, pipelines):
    '''
    Update the optimization dictionary with the best AUC, CNN, and model pipeline for each algorithm.
    :param auc_storage: Dictionary with AUC values for each model configuration.
    :param model_cnn: CNN object.
    :param optimization: Dictionary storing the best settings for each algorithm (RF, LR, XGB).
    :param pipelines: Dictionary with all model pipelines.
    :return: Updated optimization dictionary.
    '''

    # Iterate over all model configurations
    for algorithm_, auc in auc_storage.items():
        algorithm = algorithm_.split('-')[0]

        # Update the optimization dictionary if the current AUC is higher
        if auc > optimization[algorithm]['auc']:
            optimization[algorithm]['auc'] = auc
            optimization[algorithm]['cnn'] = model_cnn
            optimization[algorithm]['pipeline'] = pipelines[algorithm_]
            optimization[algorithm]['modelname'] = algorithm_

    return optimization

def store_optimal_model_settings(optimization_emr, optimization_pipelines, df, repeat=0, fold=0):
    '''
    Store information about the optimal model settings for the EMR feature set.
    :param optimization_emr: Dictionary with auc values for each model configuration.
    :param optimization_pipelines: Dictionary with all model pipelines.
    :param df: DataFrame to store the optimal settings.
    :param repeat: Repeat number (int).
    :param fold: Fold number (int).
    :return: Updated DataFrame and dictionary with optimal pipelines.
    '''

    # Initialize dictionaries to store the best AUCs, pipelines, and model names
    aucs = {'LR': 0, 'RF': 0, 'XGB': 0}
    optimal_pipelines = {'LR': None, 'RF': None, 'XGB': None}
    optimal_modelnames = {'LR': '', 'RF': '', 'XGB': ''}

    # Iterate over all model configurations
    for algorithm_, values in optimization_emr.items():
        algorithm = algorithm_.split('-')[0]
        mean_value = np.mean(values)

        # Update the optimal settings if the current mean AUC is higher
        if mean_value > aucs[algorithm]:
            aucs[algorithm] = mean_value
            optimal_pipelines[algorithm] = optimization_pipelines[algorithm_]
            optimal_modelnames[algorithm] = algorithm_

    # Store the optimal settings in the DataFrame
    for algorithm, modelname in optimal_modelnames.items():
        df = pd.concat([df, pd.DataFrame({'feature_set': 'emr', 'algorithm': algorithm, 'modelname': modelname,
                      'repeat': repeat, 'fold': fold}, index=[0])], ignore_index=True)

    return df, optimal_pipelines

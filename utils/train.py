from sklearn.utils.class_weight import compute_sample_weight
from utils.data_processing import *
from utils.train_parallel import train
from multiprocessing import Pool

# Create a multiprocessing pool with 4 processes
pool = Pool(processes=4)

def train_models(df=pd.DataFrame(), pipelines=None, outerloop=False, feature_importance=None, repeat=0, fold=0,
                 feature_set='emr_sensor', random_state=42):
    '''
    Function to train models in parallel.
    :param df: DataFrame containing features (X) and labels (y).
    :param pipelines: Dictionary of model pipelines to train, with each configuration as a key-value pair.
    :param outerloop: Boolean indicating whether the model is trained in the outer loop (to store feature importances).
    :param feature_importance: Dictionary to store feature importance for each trained model (used when outerloop=True).
    :param feature_set: Feature set type ('emr', 'emr_sensor', or 'sensor').
    :param repeat: Integer indicating the CV repeat.
    :param fold: Integer indicating the CV fold (or loop).
    :return: Trained models stored in a dictionary. If outerloop=True, also returns feature importances.
    '''

    # Initialize a dictionary to store trained models
    models = {key: {} for key in pipelines}

    # Remove instances with neutral labels and drop rows containing NaN
    df = df.loc[df['label'] != 0, :]
    df = df.dropna(axis=0)

    # Drop sensor features if the feature set is 'emr'
    if feature_set == 'emr':
        df = df.drop(columns=df.filter(regex='HR|RR|Act|Pos_|nn_feature').columns)

    # Split the DataFrame into features (X) and labels (y)
    X, y = get_Xy(df)

    # Prepare parameters for parallel model training
    models_map_params = []
    for key, value in pipelines.items():

        # Get EMR feature sets based on the pipeline type
        if 'RF' in key or 'XGB' in key:
            feature_sets = value.named_steps['preselection'].feature_sets
        else:
            feature_sets = value.named_steps['feature_selection'].feature_sets

        # Remove NaN and duplicates based on feature sets
        X_,  y_ = draw_df(X, y, feature_sets)

        # If the feature set is 'sensor', remove EMR features
        if feature_set == 'sensor':
            columns = [col for col in X_.columns if not any(sub in col for sub in ['Set1', 'Set2', 'Set3', 'Set4'])]
            X_ = X_[columns]

        # Calculate sample weights for balanced training
        sample_weights = compute_sample_weight(class_weight='balanced', y=X_.reset_index('subject')['subject'].values)

        # Store parameters for parallel training
        model_params = np.asarray([value, X_, y_, sample_weights, key, outerloop, random_state])
        models_map_params.append(model_params)

    # Train models in parallel using the multiprocessing pool
    models_pl = pool.starmap(train, models_map_params)

    # Store trained models and feature importances (if outerloop=True)
    for i in range(len(pipelines)):
        algorithm = list(pipelines.keys())[i]
        models[algorithm] = models_pl[i]['model']

        # also extract and store feature importance for models trained in the outer cv loop
        if outerloop:
            feature_importance[algorithm] = update_feature_importance(feature_importance=feature_importance[algorithm],
                                                                      model=models_pl[i], fold=fold, repeat=repeat)

    if outerloop:
        return models, feature_importance
    else:
        return models

def train_optimized_model(df=pd.DataFrame(), pipeline=None, algorithm='RF', feature_importance=pd.DataFrame(), fold=0,
                          repeat=0, feature_set='emr', random_state=42):
    '''
    Function to train a single optimized model.
    :param df: DataFrame containing features (X) and labels (y).
    :param pipeline: Model pipeline to train.
    :param algorithm: Algorithm type ('LR', 'RF', or 'XGB').
    :param feature_importance: DataFrame to store feature importances.
    :param fold: Integer indicating the CV fold (or loop).
    :param repeat: Integer indicating the CV repeat.
    :param feature_set: Feature set type ('emr', 'emr_sensor', or 'sensor').
    :return: Trained model and updated feature importances.
    '''

    # Remove instances with neutral labels and drop rows containing NaN
    df = df.loc[df['label'] != 0, :]
    df = df.dropna(axis=0)

    # Drop sensor features if the feature set is 'emr'
    if feature_set == 'emr':
        df = df.drop(columns=df.filter(regex='HR|RR|Act|Pos_|nn_feature').columns)

    # Split the DataFrame into features (X) and labels (y)
    X, y = get_Xy(df)

    # Remove duplicates based on feature sets
    if algorithm == 'RF' or algorithm == 'XGB':
        feature_sets = pipeline.named_steps['preselection'].feature_sets
    else:
        feature_sets = pipeline.named_steps['feature_selection'].feature_sets
    X_, y_ = draw_df(X, y, feature_sets)

    # Compute sample weights for balanced training
    sample_weights = compute_sample_weight(class_weight='balanced', y=X_.reset_index('subject')['subject'].values)

    # If the feature set is 'sensor', remove EMR features
    if feature_set == 'sensor':
        columns = [col for col in X_.columns if not any(sub in col for sub in ['Set1', 'Set2', 'Set3', 'Set4'])]
        X_ = X_[columns]

    # Train the model
    model = train(pipeline=pipeline, X=X_, y=y_, sample_weights=sample_weights, key=algorithm, feature_importance=True,
                  random_state=random_state)

    # Extract and update feature importances
    feature_importance = update_feature_importance(
        feature_importance=feature_importance,
        model=model, fold=fold, repeat=repeat)

    return model['model'], feature_importance

def update_feature_importance(feature_importance, model, fold, repeat):
    '''
    Update the feature importance dictionary with the latest model's feature importances.
    :param feature_importance: Dictionary to store variable importances.
    :param model: Trained model containing feature importances and feature names.
    :param fold: Integer indicating the fold number of the current CV loop.
    :param repeat: Integer indicating the repeat number of the current CV repeat.
    :return: Updated feature importance dictionary.
    '''

    # Get feature names and their importances
    feature_names = model['feature_names']
    feature_importance_values = model['feature_imp']

    # Create a DataFrame to store feature importances
    df = pd.DataFrame({feature_names[i]: feature_importance_values[i] for i in range(len(feature_names))},
                           index=[0])

    # Add fold and repeat information to the DataFrame
    df['fold'] = fold
    df['repeat'] = repeat

    # Concatenate the new DataFrame with the existing feature importance DataFrame
    feature_importance = pd.concat([feature_importance, df], ignore_index=True)

    return feature_importance

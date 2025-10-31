#function for training models used for parallel processing.
def train(pipeline='', X=None, y=None, sample_weights=None, key=None, feature_importance=False, random_state=42):
    '''
    Function that can be used in parallel processing to train a model and optionally store feature importances.
    :param pipeline: Pipeline of the model to train.
    :param X: Feature data (input).
    :param y: Target labels.
    :param sample_weights: Array of sample weights, same length as y.
    :param key: The key (or name) of the current model (or pipeline).
    :param feature_importance: Boolean to indicate whether feature importances should be extracted.
    :param random_state: Random state for reproducibility.
    :return: Dictionary containing the trained model and optionally feature importances.
    '''

    # Initialize a dictionary to store the trained model and feature importances (if applicable)
    model = {}

    try:
        # Set the random state for the classification step in the pipeline
        pipeline.set_params(classification__random_state=random_state)

        # Handle specific configurations for XGBoost models
        if 'XGB' in key:
            # Convert negative labels (-1) to 0 for binary classification
            y[y==-1] = 0

            # Set scale_pos_weight and base_score for handling class imbalance
            pipeline.set_params(classification__scale_pos_weight=len(y[y==0]) / len(y[y==1]), classification__base_score=len(y[y==1]) / len(y))

            # Train the pipeline with sample weights
            model['model'] = pipeline.fit(X, y, **{'classification__sample_weight': sample_weights})

        # Handle Random Forest models
        elif 'RF' in key:
            # Train the pipeline with sample weights
            model['model'] = pipeline.fit(X, y, **{'classification__sample_weight': sample_weights})

        # Handle Logistic Regression models
        elif 'LR' in key:
            # Train the pipeline with sample weights for both classification and feature selection steps
            model['model'] = pipeline.fit(X, y, **{'classification__sample_weight': sample_weights,
                                        'feature_selection__sample_weight': sample_weights})

        # Extract feature importances if requested
        if feature_importance:

            # For RF and XGB, get feature names and their importances
            if 'RF' in key or 'XGB' in key:
                model['feature_names'] = model['model'].named_steps['preselection'].get_feature_names_out()
                model['feature_imp'] = model['model'].named_steps['classification'].feature_importances_

            # For LR, get feature names and their coefficients
            elif 'LR' in key:
                model['feature_names'] = model['model'].named_steps['feature_selection'].get_feature_names_out()
                model['feature_imp'] = model['model'].named_steps['classification'].coef_[0]

    except ValueError as e:
        # Handle any errors during training
        print("Caught an error:", e)
        raise

    return model

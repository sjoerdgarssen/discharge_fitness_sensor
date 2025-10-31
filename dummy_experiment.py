import pandas as pd
import numpy as np

from data.generate_data import *
from utils.assess_performance import *
from utils.cross_validation import *
from utils.test import *

rs = np.random.RandomState(42)

# get data
emr_df, sensor_df = get_data(100)

# apply experiment
metrics, all_predictions, all_feature_importances, model_optimization = nestedCV(df_sensor=sensor_df,
                                                df_features=emr_df, random_state=rs, k=5, repeats=2)

# calculate performance metrics for each algorithm-feature set combination
metrics_outerloop = pd.DataFrame()
for feature_set in ['emr', 'emr_sensor', 'sensor']:
    for algorithm in ['LR', 'RF', 'XGB']:
        df = all_predictions[feature_set][algorithm]
        for repeat in df['repeat'].unique():
            for fold in df['fold'].unique():
                df_ = df.loc[(df['repeat'] == repeat) & (df['fold'] == fold), :]
                df_['label'] = df_['y']
                test_predictions = {algorithm: df_}
                df_.reset_index(inplace=True)
                df_ground_truth = get_patient_label(df_)
                metrics = assess_patient_level_performance(test_predictions=test_predictions,
                                                           df_ground_truth=df_ground_truth, outerloop=True)

                # store metrics
                metrics_outerloop = pd.concat(
                    [metrics_outerloop, pd.DataFrame({'algorithm': algorithm, 'feature_set': feature_set,
                                                      'auc': metrics[algorithm]['auc'],
                                                      'sensitivity': metrics[algorithm]['sensitivity'],
                                                      'specificity': metrics[algorithm]['specificity'],
                                                      'threshold': metrics[algorithm]['threshold'],
                                                      'repeat': repeat, 'fold': fold},
                                                     index=[0])], ignore_index=True)

# calculate mean and 95%-CI for each algorithm-feature set combination
results = assess_differences_mean_ci(metrics_outerloop)

# get feature correlation statistics of sensor features with EMR for EMR+sensor RF models
df_feat_imp = all_feature_importances['emr_sensor']['RF']
df_pred = all_predictions['emr_sensor']['RF']
result_stats, result_sensor = sensor_stats_correlation(df_pred=df_pred, df_imp=df_feat_imp)

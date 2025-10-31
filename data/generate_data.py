import pandas as pd
import numpy as np

def get_emr_features():
    features = [
        'AF_Set4_relative', 'AF_Set4_categorical', 'AF_Set4_observed_value', 'ALAT_Set4_relative',
        'ALAT_Set4_categorical', 'ALAT_Set4_observed_value',
        'ASAT_Set4_relative', 'ASAT_Set4_categorical', 'ASAT_Set4_observed_value', 'Age_Set1', 'BMI_Set1',
        'BP_dias_Set2_relative',
        'BP_dias_Set2_observed_value', 'BP_mean_Set2_relative', 'BP_mean_Set2_observed_value',
        'BP_sys_Set2_categorical', 'BP_sys_Set2_relative',
        'BP_sys_Set2_observed_value', 'Bilirubin_Total_Set4_relative', 'Bilirubin_Total_Set4_categorical',
        'Bilirubin_Total_Set4_observed_value',
        'CRP_Set4_relative', 'CRP_Set4_categorical', 'CRP_Set4_observed_value', 'Calcium_Set4_relative',
        'Calcium_Set4_categorical',
        'Calcium_Set4_observed_value', 'Chloride_Set4_relative', 'Chloride_Set4_categorical',
        'Chloride_Set4_observed_value',
        'Consciousness_Set2_categorical', 'Consciousness_Set2_relative', 'Consciousness_Set2_observed_value',
        'Creatinin_Set4_relative',
        'Creatinin_Set4_categorical', 'Creatinin_Set4_observed_value', 'Erythrocytes_Set4_relative',
        'Erythrocytes_Set4_categorical',
        'Erythrocytes_Set4_observed_value', 'Fall_risk_Set3_relative', 'Fall_risk_Set3_measured',
        'Fall_risk_Set3_observed_value',
        'GFR_Set4_relative', 'GFR_Set4_categorical', 'GFR_Set4_observed_value', 'Gender_Set1', 'Glucose_Set4_relative',
        'Glucose_Set4_categorical', 'Glucose_Set4_observed_value', 'Glucose_Urine_Set4_relative',
        'Glucose_Urine_Set4_categorical',
        'Glucose_Urine_Set4_observed_value', 'HCO3_Set4_relative', 'HCO3_Set4_categorical', 'HCO3_Set4_observed_value',
        'Hb_Set4_relative',
        'Hb_Set4_categorical', 'Hb_Set4_observed_value', 'Hematocrit_Set4_relative', 'Hematocrit_Set4_categorical',
        'Hematocrit_Set4_observed_value',
        'KATZ_Set3_relative', 'KATZ_Set3_measured', 'KATZ_Set3_observed_value', 'LD_Set4_relative',
        'LD_Set4_categorical', 'LD_Set4_observed_value',
        'Lactate_Set4_relative', 'Lactate_Set4_categorical', 'Lactate_Set4_observed_value', 'Length_Set1',
        'Leukocytes_Blood_Set4_relative',
        'Leukocytes_Blood_Set4_categorical', 'Leukocytes_Blood_Set4_observed_value', 'Leukocytes_Urine_Set4_relative',
        'Leukocytes_Urine_Set4_categorical', 'Leukocytes_Urine_Set4_observed_value', 'Lipase_Set4_relative',
        'Lipase_Set4_categorical',
        'Lipase_Set4_observed_value', 'MEWS_Combi_Set2_categorical', 'MEWS_Combi_Set2_relative',
        'MEWS_Combi_Set2_observed_value',
        'Oxygensaturation_Set2_categorical', 'Oxygensaturation_Set2_relative', 'Oxygensaturation_Set2_observed_value',
        'Oxygensupply_Set2_categorical', 'Oxygensupply_Set2_relative', 'Oxygensupply_Set2_observed_value',
        'Potassium_Set4_relative',
        'Potassium_Set4_categorical', 'Potassium_Set4_observed_value', 'Prepurse_Set3_relative',
        'Prepurse_Set3_measured',
        'Prepurse_Set3_observed_value', 'Pulserate_Set2_categorical', 'Pulserate_Set2_relative',
        'Pulserate_Set2_observed_value',
        'Respirationrate_Set2_categorical', 'Respirationrate_Set2_relative', 'Respirationrate_Set2_observed_value',
        'SNAQ_Set3_relative',
        'SNAQ_Set3_measured', 'SNAQ_Set3_observed_value', 'Sodium_Set4_relative', 'Sodium_Set4_categorical',
        'Sodium_Set4_observed_value',
        'Temperature_Set2_categorical', 'Temperature_Set2_relative', 'Temperature_Set2_observed_value',
        'Thrombocytes_Set4_relative',
        'Thrombocytes_Set4_categorical', 'Thrombocytes_Set4_observed_value', 'Urea_Set4_relative',
        'Urea_Set4_categorical',
        'Urea_Set4_observed_value', 'VAS_Set3_relative', 'VAS_Set3_measured', 'VAS_Set3_observed_value', 'Weight_Set1',
        'pCO2_Set4_relative',
        'pCO2_Set4_categorical', 'pCO2_Set4_observed_value', 'pH_Arterial_Set4_relative',
        'pH_Arterial_Set4_categorical',
        'pH_Arterial_Set4_observed_value', 'pH_Urine_Set4_relative', 'pH_Urine_Set4_categorical',
        'pH_Urine_Set4_observed_value',
        'pO2_Set4_relative', 'pO2_Set4_categorical', 'pO2_Set4_observed_value', 'yGT_Set4_relative',
        'yGT_Set4_categorical', 'yGT_Set4_observed_value'
    ]
    return features

def generate_dummy_emr_dataset(n_subjects: int) -> pd.DataFrame:
    '''
    Generate a dummy dataset with a MultiIndex (subject, timestamp), features from N(0,1),
    and a random missing fraction (10% to 30%) for each subject. Each subject has a random
    number of timestamps (between 20 and 48).

    Parameters:
        n_subjects (int): Number of unique subjects (S01, S02, ...)

    Returns:
        pd.DataFrame: Dummy dataset with MultiIndex and missing values
    '''

    # EMR feature names
    columns = get_emr_features()

    # Generate subject IDs
    subject_ids = [f'S{str(i).zfill(2)}' for i in range(1, n_subjects + 1)]

    # Initialize an empty DataFrame
    all_data = []

    for subject in subject_ids:
        # Randomly choose the number of timestamps for this subject
        n_timepoints = np.random.randint(20, 49)
        timestamps = pd.date_range(
            start='2025-01-01 18:00:00',
            periods=n_timepoints,
            freq='H'
        )

        # Generate data for this subject
        data = np.random.normal(loc=0, scale=1, size=(n_timepoints, len(columns)))

        # Assign a random missing fraction for this subject
        missing_fraction = np.random.uniform(0.25, 0.75)
        mask = np.random.rand(*data.shape) < missing_fraction
        data[mask] = np.nan

        # Create a DataFrame for this subject
        subject_df = pd.DataFrame(data, index=timestamps, columns=columns)
        subject_df['subject'] = subject
        all_data.append(subject_df)

    # Concatenate all subject data
    df = pd.concat(all_data)

    # Set MultiIndex and forward fill
    df.index.name = 'ts'
    df = df.set_index('subject', append=True)
    df = df.groupby('subject').ffill()

    return df

def generate_sensor_data(emr_df):
    '''
    Generates sensor data at a 15-minute frequency and computes derived features, which are added to the provided EMR dataframe.

    Parameters:
        emr_df (pd.DataFrame): A dataframe containing EMR data with a MultiIndex (timestamp, subject).

    Returns:
        tuple:
            pd.DataFrame: The updated EMR dataframe with additional derived features.
            pd.DataFrame: A new dataframe containing the generated sensor data at 15-minute intervals.
    '''

    # Define sensor columns
    sensor_columns = [
        'Act', 'HR', 'RR', 'Pos_lying_on_side', 'Pos_reclined_leaning',
        'Pos_supine', 'Pos_upright', 'Pos_other'
    ]

    # Initialize sensor data
    sensor_data = []

    # Process each subject
    for subject in emr_df.index.get_level_values('subject').unique():
        
        # Get the time range for the subject
        subject_data = emr_df.loc[(slice(None), subject), :]
        start_time, end_time = subject_data.index.get_level_values('ts').min(), subject_data.index.get_level_values(
            'ts').max()
        
        # Generate 15-minute frequency timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq='15T')

        # Generate random data for sensor columns
        data = {
            'Act': np.clip(np.random.normal(loc=4, scale=2, size=len(timestamps)), 0, 10),
            'HR': np.random.randint(60, 100, size=len(timestamps)),
            'RR': np.random.randint(12, 20, size=len(timestamps)),
            'Pos_lying_on_side': np.random.choice([0, 0.33, 0.67, 1], size=len(timestamps)),
            'Pos_reclined_leaning': np.random.choice([0, 0.33, 0.67, 1], size=len(timestamps)),
            'Pos_supine': np.random.choice([0, 0.33, 0.67, 1], size=len(timestamps)),
            'Pos_upright': np.random.choice([0, 0.33, 0.67, 1], size=len(timestamps)),
            'Pos_other': np.random.choice([0, 0.33, 0.67, 1], size=len(timestamps)),
        }

        # Create a DataFrame for the subject
        subject_sensor_df = pd.DataFrame(data, index=timestamps)
        subject_sensor_df['subject'] = subject

        # Add missing values (10% of rows, all features missing at the same rows)
        row_mask = np.random.rand(len(timestamps)) < 0.1
        subject_sensor_df.loc[row_mask, sensor_columns] = np.nan

        # Add additional missing values (10% of rows, only HR and RR missing)
        hr_rr_mask = (np.random.rand(len(timestamps)) < 0.1) & ~row_mask
        subject_sensor_df.loc[hr_rr_mask, ['HR', 'RR']] = np.nan

        # Append to the sensor data
        sensor_data.append(subject_sensor_df)

    # Concatenate all sensor data
    sensor_df = pd.concat(sensor_data)
    sensor_df.index.name = 'ts'
    sensor_df = sensor_df.set_index('subject', append=True)

    # generate statistical features for HR, RR and activity
    for feature in ['Act', 'HR', 'RR']:
        for subject in emr_df.index.get_level_values('subject').unique():
            subject_sensor_data = sensor_df.loc[(slice(None), subject), feature]
            first_time = subject_sensor_data.index.get_level_values('ts').min() + pd.Timedelta(hours=8)
            for ts in subject_sensor_data.index.get_level_values('ts'):
                if ts < first_time:
                    continue  # Skip timestamps before 8 hours after the first timestamp
                window_data = subject_sensor_data.reset_index('subject').loc[:ts].last('8h')
                if window_data.count().sum() >= 3:  # Ensure at least 3 non-NA values
                    emr_df.loc[(ts, subject), f'{feature}_Set5__kurtosis_8h'] = window_data[feature].kurt()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__max_8h'] = window_data[feature].max()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__mean_8h'] = window_data[feature].mean()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__min_8h'] = window_data[feature].min()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__range_8h'] = window_data[feature].max() - window_data[feature].min()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__sd_8h'] = window_data[feature].std()
                    emr_df.loc[(ts, subject), f'{feature}_Set5__skewness_8h'] = window_data[feature].skew()

    # generate statistical features for Posture variables
    for pos_feature in ['Pos_lying_on_side', 'Pos_reclined_leaning', 'Pos_supine', 'Pos_upright', 'Pos_other']:
        for subject in emr_df.index.get_level_values('subject').unique():
            subject_sensor_data = sensor_df.loc[(slice(None), subject), pos_feature]
            first_time = subject_sensor_data.index.get_level_values('ts').min() + pd.Timedelta(hours=8)
            for ts in subject_sensor_data.index.get_level_values('ts'):
                if ts < first_time:
                    continue  # Skip timestamps before 8 hours after the first timestamp
                window_data = subject_sensor_data.reset_index('subject').loc[:ts].last('8h')
                if window_data.count().sum() >= 3:  # Ensure at least 3 non-NA values
                    emr_df.loc[(ts, subject), f'{pos_feature}_Set5__mean_8h'] = window_data[pos_feature].mean()

    # forward fill
    emr_df = emr_df.groupby('subject').ffill()

    return emr_df, sensor_df

def label_data(emr_df, sensor_df):
    '''
    Labels the data with -1, 0, or 1 based on the specified criteria and applies the same labels to both EMR and sensor data.

    Parameters:
        emr_df (pd.DataFrame): The EMR dataframe with a MultiIndex (timestamp, subject).
        sensor_df (pd.DataFrame): The sensor dataframe with a MultiIndex (timestamp, subject).

    Returns:
        tuple:
            pd.DataFrame: The updated EMR dataframe with a new column 'label'.
            pd.DataFrame: The updated sensor dataframe with a new column 'label'.
    '''

    # Get unique subjects
    subjects = emr_df.index.get_level_values('subject').unique()

    # Randomly select 65% of subjects to have label -1
    n_negative = int(len(subjects) * 0.65)
    negative_subjects = np.random.choice(subjects, size=n_negative, replace=False)

    # Initialize label columns
    emr_df['label'] = -1
    sensor_df['label'] = -1

    # Assign labels for the remaining 35% of subjects
    positive_subjects = set(subjects) - set(negative_subjects)
    for subject in positive_subjects:
        # Get the last timestamp for the subject in EMR data
        subject_data = emr_df.loc[(slice(None), subject), :]
        last_time = subject_data.index.get_level_values('ts').max()
        last_12_hours = last_time - pd.Timedelta(hours=12)

        # Assign labels in EMR data
        # Temporarily reset the index
        emr_df_reset = emr_df.reset_index()

        # Assign labels in EMR data
        emr_df_reset.loc[(emr_df_reset['ts'] >= last_12_hours) & (emr_df_reset['ts'] <= last_time) & (
                        emr_df_reset['subject'] == subject), 'label'] = 1
        emr_df_reset.loc[(emr_df_reset['ts'] < last_12_hours) & (emr_df_reset['subject'] == subject), 'label'] = 0

        # Set the index back to MultiIndex
        emr_df = emr_df_reset.set_index(['ts', 'subject'])

        # Assign labels in sensor data
        # Temporarily reset the index
        sensor_df_reset = sensor_df.reset_index()

        # Assign labels in EMR data
        sensor_df_reset.loc[(sensor_df_reset['ts'] >= last_12_hours) & (sensor_df_reset['ts'] <= last_time) & (
                        sensor_df_reset['subject'] == subject), 'label'] = 1
        sensor_df_reset.loc[(sensor_df_reset['ts'] < last_12_hours) & (sensor_df_reset['subject'] == subject),
            'label'
        ] = 0

        # Set the index back to MultiIndex
        sensor_df = sensor_df_reset.set_index(['ts', 'subject'])

    return emr_df, sensor_df


def get_data(n_subjects=10):
    '''
    Generates labeled EMR and sensor data for a given number of subjects.

    :param n_subjects: int, optional
        The number of subjects for which to generate the data. Default is 10.
    :return: tuple
        A tuple containing:
        - labeled_emr_df (pd.DataFrame): The labeled EMR dataframe.
        - labeled_sensor_df (pd.DataFrame): The labeled sensor dataframe.
    '''

    # Generate dummy EMR dataset
    emr_df = generate_dummy_emr_dataset(n_subjects)

    # Generate sensor data and add features to EMR data
    updated_emr_df, sensor_df = generate_sensor_data(emr_df)

    # Add labels to both dataframes
    labeled_emr_df, labeled_sensor_df = label_data(updated_emr_df, sensor_df)

    # reset indices
    labeled_emr_df.reset_index(inplace=True)
    labeled_sensor_df.reset_index(inplace=True)

    return labeled_emr_df, labeled_sensor_df

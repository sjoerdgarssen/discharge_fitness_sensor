import numpy as np
import pandas as pd

def get_patient_label(df):
    '''
    Function to generate patient-level labels based on individual timepoint labels.
    :param df: A DataFrame with feature data that includes the columns 'subject' and 'label'.
    :return: A DataFrame with the column 'patient_y' indicating patient-level labels.
    '''

    # Get unique patient IDs and initialize a DataFrame to store patient-level labels
    subs = np.unique(df['subject'])
    df_subs = pd.DataFrame({'subject': subs, 'patient_y': np.zeros(len(subs)) * np.nan})

    # Iterate over all patients to determine their patient-level label
    for sub in subs:
        if 1 in df['label'].values[df['subject'] == sub]:
            df_subs.loc[df_subs['subject'].values == sub, 'patient_y'] = 1
        elif -1 in df['label'].values[df['subject'] == sub]:
            df_subs.loc[df_subs['subject'].values == sub, 'patient_y'] = -1
        else:
            df_subs.loc[df_subs['subject'].values == sub, 'patient_y'] = 0

    return df_subs

def get_sequences_nn(df=pd.DataFrame(), train=True):
    '''
    Function to extract sequences of sensor data and corresponding labels, patient IDs, and timestamps.
    :param df: Resampled sensor DataFrame including the columns 'label', 'ts', 'subject', and sensor features.
    :param train: Boolean indicating whether the sequence is for training (filters out neutral labels if True).
    :return: Sequences, label sequences, patient ID sequences, and timestamp sequences as numpy arrays.
    '''

    # Initialize lists to store sequences, labels, patient IDs, and timestamps
    sequences = []
    labels = []
    patients = []
    ts_seq = []

    # Define the columns containing sensor features
    cols = ['HR', 'RR', 'Act', 'Pos_lying_on_side', 'Pos_reclined_leaning', 'Pos_upright', 'Pos_supine', 'Pos_other']

    # Minimum number of instances required for sequence extraction (32 measurements for an 8-hour window)
    min_ts = 32

    # Iterate over each subject
    for subject_id in df['subject'].unique():

        # Get data for the current subject and reset the index
        subject_df = df[df['subject'] == subject_id]
        subject_df.reset_index(inplace=True, drop=True)

        # If training, filter out neutral labels and forward-fill missing labels
        if train:
            subject_df['label'] = subject_df['label'].fillna(method='ffill')
            subject_df = subject_df.loc[(subject_df['label'] != 0) & (~subject_df['label'].isna()), :]
            subject_df.loc[subject_df['label'] == -1, 'label'] = 0
            subject_df.reset_index(inplace=True, drop=True)

        # Iterate over timestamps, starting from the 32nd (index 31)
        for i in range(min_ts - 1, subject_df.shape[0]):

            # Only extract sequences at whole hours
            if subject_df['ts'][i].minute > 0:
                continue

            # Check if there are enough valid HR and RR measurements
            start_index = i - min_ts + 1
            valid_hr = subject_df.loc[start_index:i, 'HR'].notna().sum()
            valid_rr = subject_df.loc[start_index:i, 'RR'].notna().sum()
            if (valid_hr >= 0.5 * min_ts) & (valid_rr >= 0.5 * min_ts):

                # get sequence and fill missing values
                sequence = subject_df[cols].values[start_index:i+1]
                df_ = pd.DataFrame(sequence)
                df_.interpolate(method='linear', axis=0, inplace=True, limit_area='inside')
                df_.ffill(inplace=True)
                df_.bfill(inplace=True)

                # Skip if the target label is NaN
                target = subject_df.iloc[i]['label']
                if np.isnan(target):
                    continue

                # Store the sequence, target, patient ID, and timestamp
                sequence = df_.values
                ts_ = subject_df.iloc[i]['ts']
                labels.append(np.array(int(target)))
                sequences.append(sequence)
                ts_seq.append([ts_])
                patients.append([subject_id])

    # Convert lists to numpy arrays
    sequences_np = np.array(sequences)
    labels_np = np.array(labels)
    labels_np = np.expand_dims(labels_np, axis=-1)  # Add batch dimension
    patients_np = np.array(patients)
    ts_np = np.array(ts_seq)

    return sequences_np, labels_np, patients_np, ts_np

def draw_df(df=None, y=None, feature_sets=[]):
    '''
    Removes duplicate rows while ensuring consistency across feature sets.
    :param df: DataFrame with at least 'subject' and 'ts' as indices.
    :param y: Labels corresponding to the DataFrame.
    :param feature_sets: List of feature sets to consider for duplicates.
    :return: DataFrame without duplicates and corresponding labels.
    '''

    # Add labels to the DataFrame to ensure consistent indices
    df['y'] = y

    # Select columns to consider for duplicates based on feature sets.
    # Set1: patient characteristics, Set2: MEWS and vital signs, Set3: nursing assessments, Set4: laboratory values.
    columns = [col for col in df.columns if any(sub in col for sub in feature_sets)]
    columns = columns + ['subject', 'y']

    # Drop rows with NaN values and remove duplicates
    df = df.dropna(axis=0).reset_index('subject').drop_duplicates(subset=columns, keep='first').set_index('subject',
                                                                                                          append=True)

    # Split the DataFrame into features (df) and labels (y)
    y = df['y'].values
    df.drop(columns=['y'], inplace=True)

    return df, y

def get_Xy(df=pd.DataFrame()):
    '''
    Splits a DataFrame into features (X) and labels (y).
    :param df: DataFrame containing features and labels.
    :return: Feature DataFrame (X) and label array (y).
    '''

    # Extract labels
    y = df['label'].values

    # Extract input features (columns starting with 'Set' or 'nn_feature')
    columns = df.filter(regex='Set|nn_feature').columns.tolist()
    X = df[columns]

    return X, y

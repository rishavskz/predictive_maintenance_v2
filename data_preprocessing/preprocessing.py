import pickle
import numpy as np
import pandas as pd


def prepare_data_train(drop_cols=True):
    dependent_var = ['RUL']
    index_columns_names = ["UnitNumber", "Cycle"]
    operational_settings_columns_names = ["OpSet" + str(i) for i in range(1, 4)]
    sensor_measure_columns_names = ["SensorMeasure" + str(i) for i in range(1, 22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names
    with open('test_data/columns.pkl', "wb") as f:
        pickle.dump(input_file_column_names, f)
    cols_to_drop = ['OpSet3', 'SensorMeasure1', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure10',
                    'SensorMeasure14',
                    'SensorMeasure16', 'SensorMeasure18', 'SensorMeasure19']

    df_train = pd.read_csv('/home/rishav/PycharmProjects/predictive_maintenance_master_2.0/data/train_FD001.txt', delim_whitespace=True, names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
    rul.columns = ['UnitNumber', 'max']
    df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    if drop_cols:
        df_train = df_train.drop(cols_to_drop, axis=1)

    return df_train


def prepare_data_test(df_test, drop_cols=True):
    cols_to_drop = ['OpSet3', 'SensorMeasure1', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure10',
                    'SensorMeasure14',
                    'SensorMeasure16', 'SensorMeasure18', 'SensorMeasure19']
    if drop_cols:
        df_test = df_test.drop(cols_to_drop, axis=1)
    return


'''
LSTM expects an input in the shape of a numpy array of 3 dimensions that's why converting train and test data 
accordingly.
'''


def gen_train(id_df, seq_length, seq_cols):
    """
        function to prepare train data into (samples, time steps, features)
        id_df = train dataframe
        seq_length = look back period
        seq_cols = feature columns
    """

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)


def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]


def gen_test(id_df, seq_length, seq_cols, mask_value):
    """
        function to prepare test data into (samples, time steps, features)
        function only returns last sequence of data for every unit
        id_df = test dataframe
        seq_length = look back period
        seq_cols = feature columns
    """
    df_mask = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    df_mask[:] = mask_value

    id_df = df_mask.append(id_df, ignore_index=True)

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    start = num_elements - seq_length
    stop = num_elements

    lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
import tensorflow.keras.backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Masking, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import History
from tensorflow.keras import callbacks


# based on data wrangling python notebook
def prepare_data(drop_cols=True):
    dependent_var = ['RUL']
    index_columns_names = ["UnitNumber", "Cycle"]
    operational_settings_columns_names = ["OpSet" + str(i) for i in range(1, 4)]
    sensor_measure_columns_names = ["SensorMeasure" + str(i) for i in range(1, 22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names

    cols_to_drop = ['OpSet3', 'SensorMeasure1', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure10',
                    'SensorMeasure14',
                    'SensorMeasure16', 'SensorMeasure18', 'SensorMeasure19']

    df_train = pd.read_csv('train_FD004.txt', delim_whitespace=True, names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
    rul.columns = ['UnitNumber', 'max']
    df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv('test_FD004.txt', delim_whitespace=True, names=input_file_column_names)

    if (drop_cols == True):
        df_train = df_train.drop(cols_to_drop, axis=1)
        df_test = df_test.drop(cols_to_drop, axis=1)

    y_true = pd.read_csv('RUL_FD004.txt', delim_whitespace=True, names=["RUL"])
    y_true["UnitNumber"] = y_true.index

    return df_train, df_test, y_true
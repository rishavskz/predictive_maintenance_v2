import pickle
import psycopg2
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing.load_data import load_data
from db_ingestion.db_operations import run_model_op
from data_preprocessing.preprocessing import gen_test

app = Flask(__name__)

connection = psycopg2.connect(user="skzdbuser01",
                              password='Sktdb123',
                              host="34.210.66.60",
                              port="5432",
                              database="predictive_maintenance")

cursor = connection.cursor()

sequence_length = 50
mask_value = 0


@app.route("/")
def hello():
    return "Hello, World!"


@app.route('/run_process', methods=['POST'])
def run_processes():
    data = request.get_json(force=True)
    model = tf.keras.models.load_model("models/" + data['machine'] + '.h5')
    table_data = load_data(connection, data['table_name'])
    y_test = table_data[['unitnumber', 'rul']]
    y_test = y_test.groupby('unitnumber').min().reset_index()
    y_test['rul'] = y_test['rul'] - 1
    feats = pickle.load(open('test_data/feats.pkl', 'rb'))
    df_test = table_data.drop(columns=['rul', 'datetime'])
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    df_test[feats] = min_max_scaler.fit_transform(df_test[feats])
    x_test = np.concatenate(list(
        list(gen_test(df_test[df_test['unitnumber'] == unit], sequence_length, feats, mask_value)) for unit in
        df_test['unitnumber'].unique()))
    preds = model.predict(x_test, verbose=0)
    model = run_model_op(cursor, preds, y_test)
    query = """Update tb_dataset_runs set status = %s where dataset_id = %s and dataset_run_id = %s"""
    cursor.execute(query, ('S', data['id'], data['run_id']))
    count = cursor.rowcount
    connection.commit()
    print(model)
    return {"status": 200, "row_updated": count}


@app.route('/run_predict', methods=['POST'])
def live_anomaly_pred():
    data = request.get_json(force=True)
    model = tf.keras.models.load_model("models/" + data['table_name'] + '.h5')
    table_data = load_data(connection, data['table_name'])
    y_test = table_data[['unitnumber', 'rul']]
    y_test = y_test.groupby('unitnumber').min().reset_index()
    y_test['rul'] = y_test['rul'] - 1
    df_test = table_data.drop(columns=['rul', 'datetime'])
    feats = pickle.load(open('test_data/feats.pkl', 'rb'))
    df_test = table_data.drop(columns=['rul', 'datetime'])
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    df_test[feats] = min_max_scaler.fit_transform(df_test[feats])
    x_test = np.concatenate(list(
        list(gen_test(df_test[df_test['unitnumber'] == unit], sequence_length, feats, mask_value)) for unit in
        df_test['unitnumber'].unique()))
    preds = model.predict(x_test, verbose=0)
    query = """INSERT INTO machine_rul (datetime, rul, machine) VALUES (%s,%s,%s) """
    cursor.execute(query, (datetime.now(), float(preds[0][0]), data['table_name']))
    count = cursor.rowcount
    connection.commit()
    return {"status": 200, "rul": int(preds[0][0])}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3336)

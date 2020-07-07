import pickle
import psycopg2
import numpy as np
import tensorflow as tf
from flask import Flask, request
from data_preprocessing.load_data import load_data
from db_ingestion.db_operations import run_model_op
from data_preprocessing.preprocessing import gen_test

app = Flask(__name__)

connection = psycopg2.connect(user="skzdbuser01",
                              password='Sktdb123',
                              host="54.212.166.61",
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
    model = tf.keras.models.load_model(data['machine'] + '.h5')
    table_data = load_data(connection, data['table_name'])
    y_test = table_data['RUL']
    df_test = table_data.drop(columns=['RUL'])
    feats = pickle.load(open('test_data/feats.pkl', 'rb'))
    x_test = np.concatenate(list(
        list(gen_test(df_test[df_test['UnitNumber'] == unit], sequence_length, feats, mask_value)) for unit in
        df_test['UnitNumber'].unique()))
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
    model = tf.keras.models.load_model(data['machine'] + '.h5')
    df = load_data(connection, data['table_name'])
    return data

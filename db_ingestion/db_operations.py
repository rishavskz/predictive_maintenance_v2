import json
from datetime import datetime
from predict import return_metrics, return_comp, return_mse


def bulk_insert_model(cursor, data, metrics, mse, comp):
    query = """INSERT INTO tb_dataset_models (dataset_id, dataset_run_id, model_name, chart_type, params, 
                created_date, updated_date) VALUES (%s,%s,%s,%s,%s,%s,%s) """
    record_to_insert = [
        (data['id'], data['run_id'], 'multiple', metrics[0]['chart_type'], json.dumps(metrics), datetime.now(),
         datetime.now()),
        (data['id'], data['run_id'], 'multiple', mse[0]['chart_type'], json.dumps(mse), datetime.now(),
         datetime.now()),
        (data['id'], data['run_id'], 'multiple', comp[0]['chart_type'], json.dumps(comp), datetime.now(),
         datetime.now())
    ]
    cursor.executemany(query, record_to_insert)
    count = cursor.rowcount
    return count


def run_model_op(cursor, preds, y_test, data):
    metrics = return_metrics(preds, y_test)
    mse = return_mse()
    comp = return_comp(preds, y_test)
    try:
        count = bulk_insert_model(cursor, data, metrics, mse, comp)
        return {'status': 200, 'records': count, 'response': 'Records added in Model'}
    except:
        return {'status': 500, 'response': 'Model Process Failed'}

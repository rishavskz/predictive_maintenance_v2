import pickle
import numpy as np
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def return_mse():
    history = pickle.load(open('test_data/history.pkl', 'rb'))
    mse = history['mse']
    val_mse = history['val_mse']
    epochs = list(range(0, len(mse)))
    return [
        {
            "chart_type": "lineplot",
            "params": {
                "trace0": {
                    "x": epochs,
                    "y": mse,
                    "name": "Train MSE",
                    "type": "line"
                },
                "trace1": {
                    "x": epochs,
                    "y": val_mse,
                    "name": "Val MSE",
                    "type": "line"
                }
            },
            "title": "MSE Plot",
            "xlabel": "epochs",
            "ylabel": "Mean Squared Error",
        }
    ]


def return_metrics(preds, y_test):
    y_test = y_test['rul']
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, preds)
    return [
        {
            "chart_type": "horizontal bar plot",
            "params": {
                "trace0": {
                    "x": ['MSE', 'RMSE', 'MAPE'],
                    "y": [mse, rmse, mape],
                    "type": "bar"
                }
            },
            "title": "Metrics Plot",
            "xlabel": "Metrics",
            "ylabel": "Values"
        }
    ]


def return_comp(preds, y_test):
    y_test = y_test['rul']
    preds = np.concatenate(preds, axis=0)
    preds = list(preds)
    preds = [float(i) for i in preds]
    print(preds)
    unit = list(range(0, len(y_test)))
    return [
        {
            "chart_type": "lineplot",
            "params": {
                "trace0": {
                    "x": unit,
                    "y": list(preds),
                    "name": "Prediction",
                    "type": "line"
                },
                "trace1": {
                    "x": unit,
                    "y": list(y_test),
                    "name": "Ground Truth",
                    "type": "line"
                }
            },
            "title": "Output Comparison",
            "xlabel": "Cycle Number",
            "ylabel": "RUL",
        }
    ]
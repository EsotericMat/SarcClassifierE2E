import xgboost as xgb
import os
import numpy as np
from typing import Any, List

def load_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst


def predict_with_model(model: Any, features: List):
    features_names = model.feature_names
    dmatrix = xgb.DMatrix(np.array([features], dtype=float), feature_names=features_names)
    pred = model.predict(dmatrix)
    if pred > .5:
        result = 1
    else:
        result = 0
    return result, pred

def map_result(result: int):
    if result == 0:
        return 'Not Sarcasm'
    elif result == 1:
        return 'Sarcasm'
    else:
        raise ValueError('Result has to be 0 or 1')

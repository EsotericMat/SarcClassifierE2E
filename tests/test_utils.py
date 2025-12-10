from sarcasm_classifier.utils.tools import read_yaml, validate_path, connect_data_dirs
from inference.inference_utils import load_model, predict_with_model, map_result
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import os
import xgboost as xgb
import pytest

class TestTools:

    def test_read_yaml(self, tmp_path):
        d = tmp_path / "test"
        d.mkdir()
        p = d / "test.yaml"
        p.write_text('key: value\nlist: [1,4,6]')
        yaml_content = read_yaml(p)
        assert yaml_content is not None
        assert yaml_content['key'] == 'value'
        assert yaml_content['list'] == [1,4,6]

    def test_validate_path(self, tmp_path):
        d = tmp_path / "test"
        assert validate_path(d)
        empty_path = Path()
        assert validate_path(empty_path)

    def test_validate_connect_data_dirs(self, tmp_path):
        d = tmp_path / "test"
        file = Path('test.yaml')
        assert connect_data_dirs(d, file) == Path(f'{d}/test.yaml')

load_booster_mock = Mock()

@pytest.fixture
def get_features():
    return [0,0,0,2]

def predict_mock(features):
    return 0.78

class MockBooster:
    def __init__(self):
        self.feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

    def load_model(self, path):
        load_booster_mock(path)

    def predict(self, features):
        return .78

@pytest.fixture
def get_model():
    model = MockXGBoostClassifier()
    return model.get_booster()

class MockXGBoostClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0])

    def get_booster(self):
        return MockBooster()

    def load_model(self, path):
        return MockBooster()

class TestInferenceTools:

    def test_load_model(self, get_model, tmp_path, monkeypatch) :
        monkeypatch.setattr(os.path, 'isfile', lambda x: True)
        monkeypatch.setattr(xgb, 'Booster', MockBooster)

        path_to_mock_file = "valid/path/to/model.bin"
        result_booster = load_model(path_to_mock_file)

        assert isinstance(result_booster, MockBooster)


    def test_predict_with_model(self, get_model, get_features, monkeypatch):
        monkeypatch.setattr(get_model, 'predict', predict_mock)
        prediction = predict_with_model(get_model, get_features)
        assert type(prediction) == tuple
        assert len(prediction) == 2
        assert prediction == (1, 0.78)

    def test_map_result(self):
        assert map_result(1) == 'Sarcasm'
        assert map_result(0) == 'Not Sarcasm'
        with pytest.raises(ValueError):
            assert map_result(2)


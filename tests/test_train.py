import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from xgboost import XGBClassifier
from sarcasm_classifier.components.train import Trainer


@pytest.fixture
def get_trainer():
    return Trainer()

@pytest.fixture
def get_classes_df():
    return pd.DataFrame({
        'label': ['sarc', 'notsarc', 'sarc'],
        'subClass': ['gen', 'hyp', 'rq']
    })

# @pytest.fixture
# def get_subclass_dfs():
#     dfa = dfb = dfc = pd.DataFrame({
#         'label': [1,1,0,0],
#         'subClass': [0,1,1,2]
#     })
#     return dfa, dfb, dfc

@pytest.fixture
def get_training_dfs():
    dfa = dfb = dfc = pd.DataFrame(
        {
            'text':['sarcasm tester', 'another text', 'just text', 'oh my, there is a text here'],
            'label':[1,1,0,0],
            'subClass':[0,1,1,2],
            'feat1':[.432, .999, .324, .3221],
            'feat2':[-.3234, .343, .083, -.111],
            'featn':[-.003, -.00004, .0504, .323]
        }
    )
    return dfa, dfb, dfc

@pytest.fixture
def get_load_train_df():
    return pd.DataFrame({
        'feature_1': [1, 2],
        'punctuations': [3, 4],
        'repeated_punctuations': [5, 6],
        'label': [0, 1]
    })


@pytest.fixture
def setup_load_train_mocks(get_trainer, monkeypatch, get_load_train_df):
    """Mocks os.path.exists and pandas.read_csv side effect."""
    # 1. Inject Config
    get_trainer.config = MockConfig()

    # 2. Mock os.path.exists
    monkeypatch.setattr(os.path, 'exists', lambda x: True)

    # 3. Return a function that creates the mock for pd.read_csv on demand
    def create_mock_csv(num_calls=3):
        mock_list = [get_load_train_df.copy() for _ in range(num_calls)]
        mock_read_csv = Mock(side_effect=mock_list)
        return mock_read_csv

    return create_mock_csv

@pytest.fixture
def get_true_and_prediction():
    return [0, 1, 0, 1], [0, 1, 1, 1]

@pytest.fixture
def get_model():
    model = MockXGBoostClassifier()
    return model.get_booster()

class MockConfig:
    def __init__(self):
        self.train_files_path = 'tests/data'
        self.confusion_matrix_artifact_file = 'tests/artifacts/cmTest.txt'
        self.models_path = 'tests/artifacts'
        self.sarc_saved_model_file = 'sarcModel.ubj'
        self.sarc_type_saved_model_file = 'sarcTypeModel.ubj'

mock_booster_save = Mock()

class MockBooster:
    def save_model(self, path):
        mock_booster_save(path)

class MockXGBoostClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0])

    def get_booster(self):
        return MockBooster()


class TestTrainer:

    """
    Tests the core training pipeline in sarcasm_classifier.components.train. Focuses on model training and validation.
    """

    def test_label_encode(self, get_trainer, get_classes_df):
        trainer = get_trainer
        output_df = trainer.label_encode(get_classes_df)
        assert output_df['label'].dtype == 'int64'
        assert output_df['subClass'].dtype == 'int64'


    def test_prepare_subclass_dataset(self, get_trainer, get_training_dfs):
        trainer = get_trainer
        train_set, test_set, validation_set = get_training_dfs
        train_set, test_set, validation_set = trainer.prepare_subclass_dataset(train_set, test_set, validation_set)
        assert train_set['label'].unique().all() == 1
        assert test_set['label'].unique().all() == 1
        assert validation_set['label'].unique().all() == 1

    def test_prepare_training_sets(self, get_trainer, get_training_dfs):
        trainer = get_trainer
        train_set, test_set, validation_set = get_training_dfs
        train_tuple, test_tuple, val_tuple = trainer.prepare_training_sets(train_set, test_set, validation_set)
        assert len(train_tuple) == 3
        assert len(test_tuple) == 3
        assert len(val_tuple) == 3
        assert train_tuple[0].shape == (4,3)
        assert test_tuple[0].shape == (4,3)
        assert val_tuple[0].shape == (4,3)
        assert train_tuple[1].shape == (4,)
        assert test_tuple[1].shape == (4,)
        assert val_tuple[1].shape == (4,)
        assert train_tuple[2].shape == (4,)
        assert test_tuple[2].shape == (4,)
        assert val_tuple[2].shape == (4,)


    def test_load_train_case_1(self, get_trainer, setup_load_train_mocks, monkeypatch):

        mock_read_csv = setup_load_train_mocks(num_calls=3)
        monkeypatch.setattr('pandas.read_csv', mock_read_csv)

        # case 1: val=True, keep_only_embeddings = False
        a, b, c = get_trainer.load_train()
        assert not a.empty
        assert not b.empty
        assert not c.empty
        assert a.shape == (2,4)
        assert b.shape == (2,4)
        assert c.shape == (2,4)

    def test_load_train_case_2(self, get_trainer, setup_load_train_mocks, monkeypatch):
        # case 2: val=True, keep_only_embeddings = True

        mock_read_csv = setup_load_train_mocks(num_calls=3)
        monkeypatch.setattr('pandas.read_csv', mock_read_csv)

        a, b, c = get_trainer.load_train(keep_only_embeddings=True)
        assert not a.empty
        assert not b.empty
        assert not c.empty
        assert a.shape == (2,2)
        assert b.shape == (2,2)
        assert c.shape == (2,2)

    def test_load_train_case_3(self, get_trainer, setup_load_train_mocks, monkeypatch):
        # case 3: val=False, keep_only_embeddings = False

        mock_read_csv = setup_load_train_mocks(num_calls=2)
        monkeypatch.setattr('pandas.read_csv', mock_read_csv)


        monkeypatch.setattr('pandas.read_csv', mock_read_csv)
        a, b, c = get_trainer.load_train(val=False)
        assert not a.empty
        assert not b.empty
        assert c.empty
        assert a.shape == (2, 4)
        assert b.shape == (2, 4)
        assert c.shape == (0, 0)

def test_generate_confusion_matrix_artifact(get_trainer, get_true_and_prediction):
    get_trainer.config = MockConfig()
    truth, prediction = get_true_and_prediction
    cm = get_trainer.generate_confusion_matrix_artifact(
            truth,
            prediction,
            get_trainer.config.confusion_matrix_artifact_file
        )
    assert type(cm) == str
    assert os.path.isfile(get_trainer.config.confusion_matrix_artifact_file)


def test_score_model(get_trainer, get_true_and_prediction):
    truth, prediction = get_true_and_prediction
    score, accuracy, precision = get_trainer.score_model(truth, prediction)
    assert type(score) == float
    assert type(accuracy) == float
    assert type(precision) == float

def test_save_model(get_trainer, get_model, monkeypatch):
    monkeypatch.setattr('xgboost.XGBClassifier', MockXGBoostClassifier.get_booster)
    get_trainer.config = MockConfig()
    get_trainer.save_model(get_model)
    mock_booster_save.assert_called_once()














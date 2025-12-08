import numpy as np
import pytest
import pandas as pd
import math
from unittest.mock import Mock
from sarcasm_classifier.components.preprocess import Preprocess

@pytest.fixture
def get_preprocess():
    return Preprocess()

class MockConfig:
    def __init__(self):
        self.data_path = 'tests/data'
        self.gen_file = 'gen.csv'
        self.hyp_file = 'hyp.csv'
        self.rq_file = 'rq.csv'
        self.test_size = 0.33
        self.val_size = 0.5
        self.random_state = 77
        self.embedding_model = None


@pytest.fixture
def get_mock_df():
    mock_df = pd.DataFrame({
        'class':['notsarc'],
        'id':[3],
        'text':['not so sarcastic text for testing']
    })
    return mock_df

@pytest.fixture
def get_expected_df():
    expected_mock_df = pd.DataFrame({
        'label':['notsarc','notsarc','notsarc'],
        'text':['not so sarcastic text for testing', 'not so sarcastic text for testing', 'not so sarcastic text for testing'],
        'subClass':['gen','hyp','rq'],
    })
    return expected_mock_df

def get_embedding_vector():
    return np.random.rand(768)


class TestPreprocess:

    """
    Tests the core data preprocessing methods in sarcasm_classifier.components.preprocess.
    Focuses on data cleaning, splitting, and external dependencies (Mocking I/O).
    """

    def test_lower_all(self, get_preprocess):
        assert get_preprocess.lower_all("HeLLo WorLd 12##") == "hello world 12##"
        assert get_preprocess.lower_all("") == ""

    def test_remove_urls(self, get_preprocess):
        assert get_preprocess.remove_urls("https://www.fun.com") == ""
        assert get_preprocess.remove_urls("") == ""

    def test_get_punc_count(self, get_preprocess):
        assert get_preprocess.get_punc_count("?!?") == 3
        assert get_preprocess.get_punc_count("") == 0

    def test_get_repeated_puncs(self, get_preprocess):
        assert get_preprocess.get_repeated_puncs("?!? hey !!!") == 2
        assert get_preprocess.get_repeated_puncs("") == 0

    def test_remove_punctuations(self, get_preprocess):
        assert get_preprocess.remove_punctuations("?!? hey !!!").strip() == 'hey'
        assert get_preprocess.remove_punctuations("").strip() == ""

    def test_load_data(self, get_preprocess, get_mock_df, get_expected_df, monkeypatch):
        mock_df = get_mock_df
        mock_read_csv = Mock(side_effect=[mock_df.copy(), mock_df.copy(), mock_df.copy()])
        monkeypatch.setattr('pandas.read_csv', mock_read_csv)

        get_preprocess.config = MockConfig()
        actual_df = get_preprocess.load_data()

        pd.testing.assert_frame_equal(
            actual_df.reset_index(drop=True),
            get_expected_df.reset_index(drop=True)
        )

    def test_split_data(self, get_preprocess, get_expected_df):
        mock_df = get_expected_df
        get_preprocess.config = MockConfig()
        rows = mock_df.shape[0]

        mock_train_df, mock_test_df, mock_val_df = get_preprocess.split_data(
            mock_df,
            validation=False
        )


        assert mock_train_df.shape[0] == round(rows * (1 - get_preprocess.config.test_size))
        assert mock_test_df.shape[0] == round(rows * get_preprocess.config.test_size)
        assert mock_val_df.shape[0] == 0

        mock_train_df, mock_val_df, mock_test_df = get_preprocess.split_data(
            mock_df,
            validation=True
        )

        assert mock_train_df.shape[0] == round(rows * (1 - get_preprocess.config.test_size) * 0.5)
        assert mock_test_df.shape[0] == round(rows * get_preprocess.config.test_size)
        assert mock_val_df.shape[0] == mock_train_df.shape[0]

    def test_embed_model(self, get_preprocess, monkeypatch):
        get_preprocess.config = MockConfig()
        monkeypatch.setattr('sentence_transformers.SentenceTransformer.encode',  get_embedding_vector)
        embedding = get_preprocess.embed_text("test")
        assert len(embedding) == 768
        assert isinstance(embedding, (list, np.ndarray))

    def test_embedding_to_columns(self, get_preprocess, get_expected_df):
        df = get_expected_df
        df['embedding'] = get_embedding_vector
        df = get_preprocess.embedding_to_columns(df)
        assert df.shape[1] == get_expected_df.shape[1] + 767


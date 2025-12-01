import os
import sys
import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sarcasm_classifier.utils.tools import connect_data_dirs, validate_path
from pandera.pandas import Column, DataFrameSchema, Check
from sklearn.model_selection import train_test_split
from configs.manager import ConfigManager
from sentence_transformers import SentenceTransformer

class Preprocess:
    """
    Load CSV, prepare it to train a model by:
    - Load
    - Clean and remove stopwords
    - Create numeric representation to the text
    - feature engineering
    """
    def __init__(self):
        self.config = ConfigManager('preprocessing').config
        self.schema = ConfigManager('schema').config
        self.model_name = self.config.embedding_model
        self.embedding_model = SentenceTransformer(self.model_name)

    @staticmethod
    def lower_all(txt: str) -> str:
        """Make all the given text in lower case"""
        return txt.lower()

    @staticmethod
    def remove_urls(txt: str) -> str:
        """Search and remove every url element from a given text"""
        pattern = re.compile(r'https\S+|www\.S+|http?:\/\/\S+', re.IGNORECASE)
        return re.sub(pattern, '', txt).strip()

    @staticmethod
    def remove_punctuations(txt: str) -> str:
        """Remove punctuations from a given text"""
        translator = str.maketrans('', '', string.punctuation)
        return txt.translate(translator)

    @staticmethod
    def get_punc_count(txt: str) -> int:
        return len([ch for ch in txt if ch in string.punctuation])

    @staticmethod
    def get_repeated_puncs(txt: str) -> int:
        patterns = re.findall(r'([!?.\']{2,})', txt)
        return len(patterns)

    @staticmethod
    def remove_other_non_words(txt_series: pd.Series) -> pd.Series:
        return txt_series.replace(to_replace=r'[^\w\s]', value='', regex=True)


    def load_data(self) -> pd.DataFrame:
        data_path = self.config.data_path
        gen_df = pd.read_csv(connect_data_dirs(data_path, self.config.gen_file))
        hyp_df = pd.read_csv(connect_data_dirs(data_path, self.config.hyp_file))
        rq_df = pd.read_csv(connect_data_dirs(data_path, self.config.rq_file))
        gen_df['subClass'] = 'gen'
        hyp_df['subClass'] = 'hyp'
        rq_df['subClass'] = 'rq'
        unified = pd.concat([gen_df, hyp_df, rq_df], axis=0)
        unified.drop('id', axis=1, inplace=True)
        unified.rename(columns={'class': 'label'}, inplace=True)
        return unified

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        schema = self.schema
        lambda_ = lambda s: s.str.len() > 0
        subclasses = schema.possibleSarcClasses.split(', ')
        validator = DataFrameSchema(
            {
                'label': Column(schema.label, Check(lambda_)),
                'text': Column(schema.text, Check(lambda_)),
                'subClass': Column(schema.subClass, Check.isin(subclasses)),
            }
        )
        try:
            return validator.validate(df)
        except Exception as e:
            print(f'Validation error: {e}')


    def split_data(self, whole_df: pd.DataFrame, validation: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_size = self.config.test_size
        train_df, test_df = train_test_split(whole_df,
                                    test_size=test_size,
                                    random_state=self.config.random_state)
        if validation:
            validation_size = self.config.val_size
            train_df, validation_df = train_test_split(train_df,
                                                       test_size=validation_size,
                                                       random_state=self.config.random_state)
            return train_df, validation_df, test_df
        return train_df, test_df, pd.DataFrame()

    def embed_text(self, txt: str) -> np.array:
        emb = self.embedding_model.encode([txt], convert_to_numpy=True)[0]
        return emb.tolist()
        # return np.array(embedding["embeddings"])[0]

    def embedding_to_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        print(f'Embedding Text using {self.model_name}')
        dataframe['embedding'] = dataframe.text.apply(self.embed_text)
        features = [f'feat{i}' for i in range(1, 769)]
        embedding_df = pd.DataFrame(dataframe['embedding'].tolist(), columns=features, index=dataframe.index)
        dataframe = pd.concat([dataframe, embedding_df], axis=1)
        dataframe.drop('embedding', axis=1, inplace=True)
        return dataframe

    def run_single_text(self, text, add_punct: bool = True):

        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        signal = []
        text = self.lower_all(text)
        n_punct = self.get_punc_count(text)
        n_punct_in_a_row = self.get_repeated_puncs(text)
        text = self.remove_punctuations(text)
        text = self.remove_urls(text)
        embedding = self.embed_text(text)
        signal.extend(embedding)
        if add_punct:
            signal.append(n_punct)
            signal.append(n_punct_in_a_row)

        return signal

    def run(self, validation=True) -> None:
        """
        Run over all the methods, read raw data, prepare training set.
        Write to files
        :return: None
        """
        # Load & Unifiy & Validate Schema
        sarcasm_df = self.load_data()
        sarcasm_df = self.validate_schema(sarcasm_df)
        assert isinstance(sarcasm_df, pd.DataFrame), 'Something wrong with sarcasm df'

        # Text preprocessing
        sarcasm_df['text'] = sarcasm_df['text'].apply(self.lower_all)
        sarcasm_df['punctuations'] = sarcasm_df['text'].apply(self.get_punc_count)
        sarcasm_df['repeated_punctuations'] = sarcasm_df['text'].apply(self.get_repeated_puncs)
        sarcasm_df['text'] = sarcasm_df['text'].apply(self.remove_punctuations)
        sarcasm_df['text'] = sarcasm_df['text'].apply(self.remove_urls)
        sarcasm_df['text'] = self.remove_other_non_words(sarcasm_df['text'])

        # Text Embeddings
        sarcasm_df = self.embedding_to_columns(sarcasm_df)

        # Train Test Split
        sarcasm_train, sarcasm_test, sarcasm_val  = self.split_data(whole_df=sarcasm_df, validation=validation)

        # Store - ensure directory exists
        output_dir = validate_path(self.config.processed_data_path)
        sarcasm_train.to_csv(connect_data_dirs(output_dir, 'train.csv'), index=False, header=True)
        sarcasm_val.to_csv(connect_data_dirs(output_dir, 'validation.csv'), index=False, header=True)
        sarcasm_test.to_csv(connect_data_dirs(output_dir, 'test.csv'), index=False, header=True)
        print(f'Datasets stored in {output_dir.absolute()}')


if __name__ == '__main__':
    cm = ConfigManager('preprocessing').config
    preprocessing = Preprocess()
    # preprocessing.run()
    signal = preprocessing.run_single_text(text="Then why can't you explain an unreflective fertility and abortion rates between 1972-1979 here in America or how the Polish fertility rate never increased after the 1993 referendum that criminalized abortion?   Prove to me when criminalizing abortion has ever decreased it's demand. So far you haven'tonly provided empty rhetoric. The pro-life movement thoroughly brainwashes it's supporters to believe such jibberish.")
    print(signal)






















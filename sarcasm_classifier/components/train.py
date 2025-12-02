import pandas as pd
import numpy as np
import os
import mlflow
import datetime
from datetime import datetime
from typing import Tuple
import optuna
import dagshub
import logging
from xgboost import XGBClassifier
from configs.manager import ConfigManager
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sarcasm_classifier.utils.tools import connect_data_dirs, validate_path

logger= logging.getLogger(__name__)
logging.getLogger("dagshub").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)

class Trainer:

    def __init__(self):
        logger.info('Initialize Model Trainer')
        self.config = ConfigManager('model-training').config

    @staticmethod
    def label_encode(df: pd.DataFrame, target_col: str = 'label', subclass_col: str = 'subClass'):
        """
        Final step fpor training: Encode the label values
        :param df: dataset
        :param target_col: The target: 'label' or 'target' or 'class'
        :param subclass_col: 'subclass', or 'sarctype' etc.
        :return:
        """
        label_renaming_map = {'sarc': 1, 'notsarc': 0}
        subclass_renaming_mape = {'gen': 0, 'hyp': 1, 'rq': 2}
        df.replace(
            {
                target_col: label_renaming_map,
                subclass_col: subclass_renaming_mape
            },
            inplace=True
        )
        return df

    @staticmethod
    def prepare_subclass_dataset(train_set, test_set, validation_set):
        if not validation_set.empty:
            return train_set[train_set.label == 1], test_set[test_set.label == 1], validation_set[
                validation_set.label == 1]
        return train_set[train_set.label == 1], test_set[test_set.label == 1], pd.DataFrame()

    def prepare_training_sets(self, train_set, test_set, validation_set=None, target: str = 'sarcasm'):
        to_drop = ['subClass', 'text', 'label']

        if target == 'subclass':
            logger.info('Target is Sarcasm subclass, Trimming datasets')
            train_set, test_set, validation_set = self.prepare_subclass_dataset(
                train_set,
                test_set,
                validation_set=validation_set,
            )

        # Train Set
        X_train = train_set.drop(to_drop, axis=1)
        y_train = train_set['label']
        ytrain_sc = train_set['subClass']

        # Test Set
        X_test = test_set.drop(to_drop, axis=1)
        y_test = test_set['label']
        ytest_sc = test_set['subClass']

        if not validation_set.empty:
            # Validation Set
            X_val = validation_set.drop(to_drop, axis=1)
            y_val = validation_set['label']
            yval_sc = validation_set['subClass']

            return (X_train, y_train, ytrain_sc), (X_test, y_test, ytest_sc), (X_val, y_val, yval_sc)
        return (X_train, y_train, ytrain_sc), (X_test, y_test, ytest_sc), (pd.DataFrame(), [], [])


    def load_train(self, val=True, keep_only_embeddings=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data sets for training purposes
        :param val: If True, load also validation set
        :param keep_only_embeddings: if False, remove punctuation count and punctuiation in a row count.
        :return:
        """
        train_files_path = self.config.train_files_path

        if os.path.exists(train_files_path):

            train_set = pd.read_csv(connect_data_dirs(train_files_path, 'train.csv'))
            test_set = pd.read_csv(connect_data_dirs(train_files_path, 'test.csv'))
            validation_set = pd.DataFrame()

            if val:
                validation_set = pd.read_csv(connect_data_dirs(train_files_path, 'validation.csv'))

            if keep_only_embeddings:
                train_set.drop(['punctuations', 'repeated_punctuations'], axis=1, inplace=True)
                test_set.drop(['punctuations', 'repeated_punctuations'], axis=1, inplace=True)
                try:
                    validation_set.drop(['punctuations', 'repeated_punctuations'], axis=1, inplace=True)
                except:
                    pass

            return train_set, test_set, validation_set

        else:
            raise FileExistsError("Preprocessed files are not avilable")

    @staticmethod
    def generate_confusion_matrix_artifact(truth, predictions, file_path, labels=None, ):

        cm = confusion_matrix(truth, predictions, labels=labels)

        header = "Confusion Matrix Report\n"
        header += f"Generated: {datetime.now()}\n"
        header += "-" * 50 + "\n\n"
        matrix_text = "Predicted →\t" + "\t".join(str(l) for l in labels) + "\n"

        # Matrix rows
        for i, label in enumerate(labels):
            row_values = "\t".join(str(v) for v in cm[i])
            matrix_text += f"Actual {label}\t{row_values}\n"

        full_text = header + matrix_text

        with open(file_path, "w") as f:
            f.write(full_text)

        return file_path

    def score_model(self, truth, predictions):
        score = f1_score(truth, predictions)
        accuracy = accuracy_score(truth, predictions)
        precision = precision_score(truth, predictions)
        return score, accuracy, precision


    def tune_xgboost_with_optuna(self,
                                 training_object: Tuple[Tuple, Tuple, Tuple],
                                 target_type: str = 'sarcasm',
                                 n_trials: int = 100,
                                 random_state: int = 42) -> dict:
        """
        Hyperparameter tuning for XGBoost using Optuna optimization.

        Parameters:
        -----------
        training_object : tuple
            A tuple containing (train_set, test_set, validation_set) where each set is
            a tuple of (X, y, y_subclass)
        target_type : str, default 'sarcasm'
            'sarcasm' for binary classification or 'subclass' for multiclass classification
        n_trials : int, default 100
            Number of optimization trials to run
        random_state : int, default 42
            Random state for reproducibility

        Returns:
        --------
        dict : Dictionary which contains the best parameters, best score, and trained model
        """

        train_set, test_set, validation_set = training_object
        X_train, y_train, y_train_sc = train_set
        X_val, y_val, y_val_sc = validation_set

        # Select target based on classification type
        if target_type == 'sarcasm':
            y_train_target = y_train
            y_val_target = y_val
            num_classes = 2
        elif target_type == 'subclass':
            y_train_target = y_train_sc
            y_val_target = y_val_sc
            num_classes = len(y_train_sc.unique())
        else:
            raise ValueError("target_type must be 'sarcasm' or 'subclass'")

        def objective(trial):
            # Suggest hyperparameters
            with mlflow.start_run():
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': random_state,
                    'verbosity': 0
                }

                # For multiclass classification
                if num_classes > 2:
                    params['objective'] = 'multi:softmax'
                    params['num_class'] = num_classes
                    params['eval_metric'] = 'mlogloss'
                else:
                    params['objective'] = 'binary:logistic'
                    params['eval_metric'] = 'logloss'

                # Create and train model
                model = XGBClassifier(**params)
                model.fit(X_train, y_train_target,
                          eval_set=[(X_val, y_val_target)],
                          verbose=False)

                # Make predictions and calculate score
                y_pred = model.predict(X_val)

                # Use F1-score for optimization (macro average for multiclass)
                cm_file_path = self.config.confusion_matrix_artifact_file
                if num_classes > 2:
                    score, accuracy, precision = self.score_model(y_val_target, y_pred)
                    cm_file_path = self.generate_confusion_matrix_artifact(y_val_target,
                                                                           y_pred,
                                                                           labels=[0,1,2],
                                                                           file_path=cm_file_path)
                else:
                    score, accuracy, precision = self.score_model(y_val_target, y_pred)
                    cm_file_path = self.generate_confusion_matrix_artifact(y_val_target,
                                                                           y_pred,
                                                                           labels=[0, 1],
                                                                           file_path=cm_file_path)

                mlflow.log_params(params)
                mlflow.log_metric('Accuracy', accuracy)
                mlflow.log_metric('F1', score)
                mlflow.log_metric('Precision', precision)
                mlflow.log_artifact(cm_file_path)


                logger.info(f"Epoch {trial.number + 1} | ValAccuracy {accuracy} | ValPrecision: {precision}", extra={'params':params})

                return score


        # Create study and optimize
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'random_state': random_state,
            'verbosity': 0
        })

        if num_classes > 2:
            best_params['objective'] = 'multi:softmax'
            best_params['num_class'] = num_classes
            best_params['eval_metric'] = 'mlogloss'
        else:
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'logloss'

        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train, y_train_target)

        # Calculate final scores
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)

        if num_classes > 2:
            train_f1 = f1_score(y_train_target, train_pred, average='macro')
            val_f1 = f1_score(y_val_target, val_pred, average='macro')
        else:
            train_f1 = f1_score(y_train_target, train_pred)
            val_f1 = f1_score(y_val_target, val_pred)

        train_acc = accuracy_score(y_train_target, train_pred)
        val_acc = accuracy_score(y_val_target, val_pred)

        return {
            'best_params': best_params,
            'best_f1_score': study.best_value,
            'model': final_model,
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'train_f1': train_f1,
            'validation_f1': val_f1,
            'study': study
        }

    def save_model(self, model: XGBClassifier, target: str = 'sarcasm') -> None:
        """
        Store the trained XGBoost object in models directory
        :param model: The trained model object
        :param target: The classifier target (sarcasm \ subclass)
        :return: None
        """
        validate_path(self.config.models_path)

        if target == 'sarcasm':
            model_path = self.config.sarc_saved_model_file
        elif target == 'subclass':
            model_path = self.config.sarc_type_saved_model_file
        else:
            raise NotImplementedError()

        model_path = connect_data_dirs(self.config.models_path, model_path)
        model.save_model(model_path)
        logger.info(f'Model Saved in {model_path}')
        return None

    def run(self, target: str = 'sarcasm'):
        try:
            logger.info('Connecting to Dagshub')
            dagshub.init(repo_owner='matanst7', repo_name='SarcClassifierE2E', mlflow=True)
        except ConnectionError as e:
            logger.warning(f"Failed to connect to Dagshub: {e}")
            logger.info('Keep tracking using Local MLFlow')
            mlflow.set_tracking_uri("file:./mlruns")

        validate_path(self.config.artifacts_dir)

        # Load
        logger.info('Loading datasets')
        train, test, validation = self.load_train(
            val=self.config.validation,
            keep_only_embeddings=self.config.keep_punctuation_features
        )

        # Encode labels
        logger.info('Label Encoding')
        train = self.label_encode(df=train, target_col='label', subclass_col='subClass')
        test = self.label_encode(df=test, target_col='label', subclass_col='subClass')
        validation = self.label_encode(df=validation, target_col='label', subclass_col='subClass')

        # Prepare training sets:
        logger.info('Preparing training sets')
        train_set, test_set, validation_set = self.prepare_training_sets(
            train_set=train,
            test_set=test,
            validation_set=validation,
            target=target
        )

        # Train a Model Using Optuna - Validation
        mlflow.set_experiment(f"Sarcasm : {target} Classifier_{datetime.now().strftime('%Y%m%d-%H')}")
        logger.info('Training a new XGBoost Model Using Optuna for Hyperparameters tuning')
        sarcasm_results = self.tune_xgboost_with_optuna(
            training_object=(train_set, test_set, validation_set),
            target_type=target,
            n_trials=self.config.num_trials,
            random_state=77
        )

        best_model = sarcasm_results['model']

        # Results:
        logger.info(f"""Training Completed\nSarcasm Classification Results:
        - Best F1 Score: {sarcasm_results['best_f1_score']:.4f}
        - Validation Accuracy: {sarcasm_results['validation_accuracy']:.4f})
        - Validation F1: {sarcasm_results['validation_f1']:.4f})
        - Best Parameters: {sarcasm_results['best_params']}""")

        self.save_model(best_model, target)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(target='sarcasm')











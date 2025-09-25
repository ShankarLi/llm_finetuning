import json
import logging
import os
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


class ProductionHyperparameterTuner:
    """Production-ready hyperparameter tuning with MLflow tracking"""

    def __init__(self, experiment_name="sentiment_analysis_optimization"):
        self.setup_logging()
        self.setup_mlflow(experiment_name)

    def setup_logging(self):
        """Setup logging for hyperparameter tuning"""
        self.logger = logging.getLogger(__name__)

    def setup_mlflow(self, experiment_name):
        """Setup MLflow for experiment tracking"""
        try:
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            self.logger.warning(
                f"MLflow setup failed: {e}. Continuing without tracking."
            )

    def objective_logistic_regression(self, trial, X_train, y_train):
        """Objective function for Logistic Regression optimization"""

        # Suggest hyperparameters
        max_features = trial.suggest_int("max_features", 1000, 10000)
        ngram_choice = trial.suggest_categorical("ngram_range", ["1_1", "1_2", "1_3"])
        min_df = trial.suggest_int("min_df", 1, 10)
        max_df = trial.suggest_float("max_df", 0.7, 1.0)
        C = trial.suggest_float("C", 0.01, 100, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        max_iter = trial.suggest_int("max_iter", 500, 2000)
        class_weight = trial.suggest_categorical("class_weight", ["None", "balanced"])

        # Convert string representations back to tuples
        ngram_map = {"1_1": (1, 1), "1_2": (1, 2), "1_3": (1, 3)}
        ngram_range = ngram_map[ngram_choice]
        class_weight_val = None if class_weight == "None" else class_weight

        params = {
            "vectorizer__max_features": max_features,
            "vectorizer__ngram_range": ngram_range,
            "vectorizer__min_df": min_df,
            "vectorizer__max_df": max_df,
            "classifier__C": C,
            "classifier__solver": solver,
            "classifier__max_iter": max_iter,
            "classifier__class_weight": class_weight_val,
        }

        # Create pipeline
        pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        pipeline.set_params(**params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        # Log to MLflow with proper error handling
        try:
            with mlflow.start_run(nested=True):
                log_params = {
                    "max_features": max_features,
                    "ngram_range": ngram_choice,
                    "min_df": min_df,
                    "max_df": max_df,
                    "C": C,
                    "solver": solver,
                    "max_iter": max_iter,
                    "class_weight": class_weight,
                }
                mlflow.log_params(log_params)
                mlflow.log_metric("cv_f1_mean", float(scores.mean()))
                mlflow.log_metric("cv_f1_std", float(scores.std()))
        except Exception as e:
            self.logger.warning(f"MLflow logging failed: {e}")

        return scores.mean()

    def optimize_model(
        self, X_train, y_train, model_type="logistic_regression", n_trials=100
    ):
        """Optimize hyperparameters for specified model type"""

        self.logger.info(
            f"Starting optimization for {model_type} with {n_trials} trials"
        )

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Select objective function
        if model_type == "logistic_regression":
            objective_func = lambda trial: self.objective_logistic_regression(
                trial, X_train, y_train
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Optimize
        study.optimize(objective_func, n_trials=n_trials, timeout=3600)

        # Convert best parameters back to proper format
        best_params_converted = self._convert_params_from_optuna(
            study.best_trial.params, model_type
        )

        # Log best results
        self.logger.info(f"Best trial: {study.best_trial.value:.4f}")
        self.logger.info(f"Best params: {best_params_converted}")

        # Save study results
        os.makedirs("output", exist_ok=True)
        study_path = f"output/optuna_study_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(study, study_path)

        return best_params_converted, study.best_trial.value

    def _convert_params_from_optuna(self, optuna_params, model_type):
        """Convert Optuna parameters back to sklearn format"""
        converted = {}

        for key, value in optuna_params.items():
            if key == "ngram_range":
                ngram_map = {"1_1": (1, 1), "1_2": (1, 2), "1_3": (1, 3)}
                converted[key] = ngram_map[value]
            elif key == "class_weight":
                converted[key] = None if value == "None" else value
            else:
                converted[key] = value

        return converted

    def train_best_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        best_params,
        model_type="logistic_regression",
    ):
        """Train the best model with optimized hyperparameters"""

        self.logger.info("Training best model with optimized hyperparameters")

        # Create pipeline with best parameters
        pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        # Map the parameter names correctly
        corrected_params = {}
        for key, value in best_params.items():
            if key in ["max_features", "ngram_range", "min_df", "max_df"]:
                corrected_params[f"vectorizer__{key}"] = value
            elif key in ["C", "solver", "max_iter", "class_weight"]:
                corrected_params[f"classifier__{key}"] = value

        pipeline.set_params(**corrected_params)

        # Train on full training set
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro")

        self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Validation F1: {val_f1:.4f}")

        # FIXED: Log final model to MLflow with proper input example
        try:
            with mlflow.start_run():
                log_params = {}
                for key, value in corrected_params.items():
                    if isinstance(value, tuple):
                        log_params[key] = str(value)
                    elif value is None:
                        log_params[key] = "None"
                    else:
                        log_params[key] = value

                mlflow.log_params(log_params)
                mlflow.log_metric("val_accuracy", float(val_accuracy))
                mlflow.log_metric("val_f1", float(val_f1))

                # FIXED: Convert pandas Series to list for input_example
                if hasattr(X_train, "iloc"):
                    input_example = X_train.iloc[:3].tolist()
                else:
                    input_example = list(X_train[:3])

                mlflow.sklearn.log_model(
                    pipeline, "sentiment_model", input_example=input_example
                )
        except Exception as e:
            self.logger.warning(f"MLflow logging failed: {e}")

        return pipeline, val_accuracy, val_f1

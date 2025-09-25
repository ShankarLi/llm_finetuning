import json
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_manager import DataManager
from hyperparameter_tuner import ProductionHyperparameterTuner


class ProductionModelTrainer:
    """Production-ready model training pipeline"""

    def __init__(self):
        self.setup_logging()
        self.data_manager = DataManager()
        self.tuner = ProductionHyperparameterTuner()

    def setup_logging(self):
        """Setup logging"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/production_training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run_full_pipeline(self, model_types=["logistic_regression"], n_trials=50):
        """Run the complete production training pipeline"""

        self.logger.info("Starting production model training pipeline")

        # 1. Load and preprocess data
        self.logger.info("Loading HuggingFace dataset...")
        dataset = self.data_manager.load_huggingface_dataset()

        train_df = dataset["train"]
        val_df = dataset["validation"]
        test_df = dataset["test"]

        # 2. Preprocess text
        self.logger.info("Preprocessing text data...")
        train_df = self.data_manager.preprocess_text(train_df)
        val_df = self.data_manager.preprocess_text(val_df)
        test_df = self.data_manager.preprocess_text(test_df)

        # 3. Encode labels
        train_df, val_df, test_df, label_encoder = self.data_manager.encode_labels(
            train_df, val_df, test_df
        )

        # 4. Generate statistics
        stats = self.data_manager.get_data_statistics(train_df, val_df, test_df)
        self.logger.info(f"Dataset statistics: {stats}")

        # 5. Prepare data for training
        X_train, y_train = train_df["cleaned_review"], train_df["sentiment"]
        X_val, y_val = val_df["cleaned_review"], val_df["sentiment"]
        X_test, y_test = test_df["cleaned_review"], test_df["sentiment"]

        # 6. Hyperparameter optimization for each model type
        best_models = {}

        for model_type in model_types:
            self.logger.info(f"Optimizing {model_type}...")

            # Optimize hyperparameters
            best_params, best_score = self.tuner.optimize_model(
                X_train, y_train, model_type=model_type, n_trials=n_trials
            )

            # Train best model
            best_model, val_acc, val_f1 = self.tuner.train_best_model(
                X_train, y_train, X_val, y_val, best_params, model_type
            )

            best_models[model_type] = {
                "model": best_model,
                "params": best_params,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "cv_score": best_score,
            }

        # 7. Select best overall model
        best_model_type = max(
            best_models.keys(), key=lambda k: best_models[k]["val_f1"]
        )
        final_model = best_models[best_model_type]["model"]

        self.logger.info(f"Best model type: {best_model_type}")

        # 8. Final evaluation on test set
        test_metrics = self.evaluate_final_model(
            final_model, X_test, y_test, label_encoder
        )

        # 9. Save final model and metadata
        model_info = self.save_final_model(
            final_model,
            best_models[best_model_type],
            test_metrics,
            label_encoder,
            stats,
        )

        self.logger.info("Production model training completed successfully!")
        return model_info

    def evaluate_final_model(self, model, X_test, y_test, label_encoder):
        """Comprehensive evaluation of the final model"""

        self.logger.info("Evaluating final model on test set...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        }

        # Classification report
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        )

        # Convert numpy types to Python types for JSON serialization
        report_converted = self._convert_numpy_types(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Save visualizations
        self.save_evaluation_plots(cm, label_encoder.classes_, metrics)

        # Save detailed results - FIXED: Convert all numpy types
        test_results = {
            "metrics": metrics,
            "classification_report": report_converted,
            "confusion_matrix": cm.tolist(),
        }

        with open("output/test_evaluation_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

        self.logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")

        return metrics

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {
                str(key): self._convert_numpy_types(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_evaluation_plots(self, cm, class_names, metrics):
        """Save evaluation plots"""

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix - Final Model")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("output/final_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Metrics bar plot
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = plt.bar(metric_names, metric_values, color="skyblue", alpha=0.8)
        plt.title("Final Model Performance Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/final_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_final_model(
        self, model, model_info, test_metrics, label_encoder, dataset_stats
    ):
        """Save the final model with comprehensive metadata"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"output/production_sentiment_model_{timestamp}.pkl"

        # Save model
        joblib.dump(model, model_filename)

        # Convert parameters to JSON-serializable format
        serializable_params = {}
        for key, value in model_info["params"].items():
            if isinstance(value, tuple):
                serializable_params[key] = list(value)
            elif value is None:
                serializable_params[key] = None
            else:
                serializable_params[key] = value

        # Create comprehensive model metadata
        model_metadata = {
            "model_filename": model_filename,
            "timestamp": timestamp,
            "model_type": "production_optimized",
            "hyperparameters": serializable_params,
            "validation_metrics": {
                "accuracy": float(model_info["val_accuracy"]),
                "f1_score": float(model_info["val_f1"]),
                "cv_score": float(model_info["cv_score"]),
            },
            "test_metrics": test_metrics,
            "dataset_statistics": dataset_stats,
            "label_encoder_classes": label_encoder.classes_.tolist(),
            "feature_extraction": "TF-IDF",
            "preprocessing_steps": [
                "URL removal",
                "Email removal",
                "Mention/hashtag removal",
                "Special character removal",
                "Tokenization",
                "Stopword removal",
                "Lemmatization",
            ],
        }

        # Save metadata
        metadata_filename = f"output/model_metadata_{timestamp}.json"
        with open(metadata_filename, "w") as f:
            json.dump(model_metadata, f, indent=2)

        self.logger.info(f"Final model saved: {model_filename}")
        self.logger.info(f"Model metadata saved: {metadata_filename}")

        return model_metadata


def main():
    """Main execution function"""
    trainer = ProductionModelTrainer()

    # Train models with optimized trial count
    model_info = trainer.run_full_pipeline(
        model_types=["logistic_regression"],
        n_trials=75,  # Balanced optimization time vs performance
    )

    print(f"\n{'='*60}")
    print("🎉 PRODUCTION MODEL TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"📁 Model saved: {model_info['model_filename']}")
    print(f"🎯 Test accuracy: {model_info['test_metrics']['accuracy']:.4f}")
    print(f"📊 Test F1 (macro): {model_info['test_metrics']['f1_macro']:.4f}")
    print(f"🔧 Validation F1: {model_info['validation_metrics']['f1_score']:.4f}")
    print(f"⚡ CV Score: {model_info['validation_metrics']['cv_score']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline


def build_and_train_model(df):
    """Build and train the sentiment analysis model"""
    logging.info("Starting model building and training")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Prepare data
    X = df["cleaned_review"]
    y = df["sentiment"]

    # Split data - modified for small datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logging.info(
        f"Data split: training set={X_train.shape[0]}, test set={X_test.shape[0]}"
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Define hyperparameters for tuning
    param_grid = {
        "vectorizer__max_features": [3000, 5000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["lbfgs", "saga"],
        "classifier__max_iter": [500, 1000],
    }

    # Cross-validation
    grid = GridSearchCV(
        pipeline, param_grid, cv=2, scoring="f1_macro", verbose=1, n_jobs=-1
    )

    # Train the model
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score = grid.best_score_
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best cross-validation score: {best_score:.4f}")

    # Evaluate on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Test accuracy: {accuracy:.4f}")
    logging.info(f"Classification report: {report}")

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("output/confusion_matrix.png")
    plt.close()

    return best_model, X_test, y_test

import logging
import os
from llm_finetuning.non_prod_training.data_analysis import load_data, explore_data
from llm_finetuning.non_prod_training.data_preparation import (
    download_nltk_resources,
    preprocess_text,
)
from llm_finetuning.non_prod_training.train_model import build_and_train_model
from llm_finetuning.non_prod_training.deploy_model import save_model, SentimentAnalyzer


def setup_logging():
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/sentiment_pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main():
    """Main pipeline execution"""
    try:
        setup_logging()
        logging.info("Starting sentiment analysis pipeline")

        # Step 1: Data Collection
        df = load_data()

        # Step 2: Download NLTK resources
        download_nltk_resources()

        # Step 3: Data Exploration
        df = explore_data(df)

        # Step 4: Data Preprocessing
        df = preprocess_text(df)

        # Step 5: Model Building and Training
        model, X_test, y_test = build_and_train_model(df)

        # Step 6: Save Model
        model_path = save_model(model)

        # Step 7: Test model with examples
        analyzer = SentimentAnalyzer(model_path)

        sample_texts = [
            "This was a fantastic experience!",
            "I really hated this product.",
            "It's somewhat okay, nothing special.",
        ]

        print("\nExample predictions:")
        for text in sample_texts:
            result = analyzer.predict(text)
            print(f"Text: '{text}'")
            print(
                f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})"
            )
            print("---")

        print(f"\nModel saved to: {model_path}")
        print("All outputs saved to 'output/' directory")
        print("Logs saved to 'logs/' directory")

        logging.info("Sentiment analysis pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

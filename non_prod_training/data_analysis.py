import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging
logging.basicConfig(
    filename="sentiment_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_data():
    """Load sample data for sentiment analysis"""
    try:
        data = {
            "review": [
                "I loved this movie, it was fantastic!",
                "Terrible film. Waste of time.",
                "It was okay, not great but not bad.",
                "Absolutely wonderful! Best movie ever.",
                "Awful. I hated every minute.",
                "The acting was good but the plot was weak.",
                "I would recommend this to everyone!",
                "Not worth the money or time spent.",
                "Mixed feelings about this one.",
                "Best experience of my life!",
            ],
            "sentiment": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
            ],
        }
        df = pd.DataFrame(data)
        logging.info(f"Data loaded successfully with {df.shape[0]} records")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise


def explore_data(df):
    """Perform exploratory data analysis"""
    logging.info("Starting exploratory data analysis")
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Visualize sentiment distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="sentiment", data=df)
    plt.title("Sentiment Distribution")
    plt.savefig("output/sentiment_distribution.png")
    plt.close()

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")

    # Review length statistics
    df["review_length"] = df["review"].apply(len)
    print(f"Review length statistics:\n{df['review_length'].describe()}")

    return df

import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
#prova
import logging
from src import config
from src.load_data import load_data
from src.preprocess import preprocess_data
from src.make_model import train_model
# from src.evaluation import evaluate_model
# from src.save_results import save_predictions

# Set up logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#prova

def main():
    logging.info("Starting Sentiment Analysis Pipeline...")

    # Step 1: Load data from Excel and store it in SQLite
    logging.info("Loading raw data...")
    load_data()

    # Step 2: Preprocess text data
    logging.info("Preprocessing data...")
    preprocess_data()

    # Step 3: Train sentiment analysis model
    logging.info("Training the model...")
    train_model()


if __name__ == "__main__":
    main()
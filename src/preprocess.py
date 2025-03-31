import os
import sys
import pandas as pd
import nltk
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from wordcloud import STOPWORDS
#import contractions
import sqlite3
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
from src import config

def preprocess_data():


    # Download necessary resources
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')).union(STOPWORDS)

    def preprocess_tweet(text):
        """Preprocesses a tweet by performing various cleaning and normalization steps."""
        if not isinstance(text, str) or text.strip() == "":
            return ""

        # Convert to lowercase
        text = text.lower()

        # Tokenize words
        words = word_tokenize(text)

        # Remove URLs
        words = [word for word in words if not urlparse(word).scheme]  # Checks if it's a URL

        # Remove mentions (@username)
        words = [word for word in words if not word.startswith('@')]

        # Expand contractions (e.g., "can't" -> "cannot")
        #words = [contractions.fix(word) for word in words]

        # Remove punctuation & special characters (keep emojis)
        words = [word for word in words if word not in string.punctuation]

        # Convert emojis to text (e.g., 😊 -> "smiling_face_with_smiling_eyes")
        words = [emoji.demojize(word).replace("_", " ") for word in words]

        # Remove stopwords
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        words = [lemmatizer.lemmatize(word) for word in words]

        # Reconstruct cleaned text
        return " ".join(words)

    

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Read a table into a Pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {config.RAW_TABLE}", conn)

    # Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_tweet)
    df['sentiment'] = df['sentiment'].apply(lambda x : x.lower())
    df.to_sql(config.PROCESSED_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    print(f'Tweets are cleaned and loaded in {config.PROCESSED_TABLE} table.')
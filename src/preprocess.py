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

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score

def preprocess_data():


    # Download necessary resources
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')).union(STOPWORDS)

    def preprocess_comments(text):
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

        # 📌 2️⃣ Pulizia e Preparazione dei Dati, ControlloS dei Missing
        # Rimuove righe con valori mancanti "essnendo testuali fare imputation diventa complesso"
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0,'neutral': 2})  # Converte sentiment in numerico
        df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x <= 0.5 else (1 if x <= 1.5 else 2))

        # Reconstruct cleaned text
        return " ".join(words)

    def imputation_missing_data():
        # Inizializziamo l'IterativeImputer (MICE)
        imputer = IterativeImputer(random_state=42)

        # Applicare l'imputer solo sulla colonna 'sentiment' che ha i valori mancanti

        df['sentiment'] = imputer.fit_transform(df[['sentiment']])
        return df


    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Read a table into a Pandas DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {config.RAW_TABLE}", conn)

    # Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_comments)
    if df.isnull().sum().sum() != 0:
        df=df.apply(imputation_missing_data)
    
    df.to_sql(config.PROCESSED_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    print(f'Tweets are cleaned and loaded in {config.PROCESSED_TABLE} table.')
from src import config
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def load_data():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT cleaned_text, sentiment FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


#def train_model(model_name="random_forest", grid_search=False):
def train_model(grid_search=False):
    """Trains a specified model and saves evaluation metrics to SQLite."""
    model_name= "random_forest"
    
    df = load_data().head(1000)

    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    with open(f"{config.MODELS_PATH}vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    # 📌 1️⃣ Selezione del modello e pre-processing specifico
    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_name == "logistic_regression":
        # Pre-processing specifico per la Regressione Logistica (Box-Cox)
        transformer = PowerTransformer(method='box-cox')
        X_train = transformer.fit_transform(X_train.toarray())  # Convertire in array prima di Box-Cox
        X_test = transformer.transform(X_test.toarray())

        model = LogisticRegression(max_iter=500)
        param_grid = {'C': [0.01, 0.1, 1, 10]}

    elif model_name == "naive_bayes":
        # Pre-processing specifico per Naive Bayes (niente trasformazioni complesse)
        model = MultinomialNB()
        param_grid = {'alpha': [0.1, 0.5, 1]}

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")

    # 📌 2️⃣ Addestramento con GridSearchCV (se richiesto)
    if grid_search:
        grid_search_cv = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search_cv.fit(X_train, y_train)
        best_model = grid_search_cv.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    # 📌 3️⃣ Generazione delle previsioni
    y_pred = best_model.predict(X_test)

    # 📌 4️⃣ Salvataggio del modello
    model_path = f"{config.MODELS_PATH}{model_name}.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)

    # 📌 5️⃣ Creazione del DataFrame con le previsioni
    test_df = df.loc[test_idx].copy()
    test_df['prediction'] = y_pred  

    # 📌 6️⃣ Calcolo delle metriche
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # 📌 7️⃣ Salvataggio nel database
    conn = sqlite3.connect(config.DATABASE_PATH)
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    logging.info(f"✅ Modello {model_name} addestrato e salvato con successo!")


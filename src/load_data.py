import sqlite3
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
from src import config

import logging
# Set up logging

def load_data():
    logging.info('Opening Excel File...')
    YT = pd.read_csv(os.path.join(config.RAW_DATA_PATH,
                    'YoutubeCommentsDataSet.csv'))
    


    df = YT[['Comment', 'Sentiment']]
    df = df.rename(columns={"Comment": "text", "Sentiment": "sentiment"})
    df = df.dropna() 
    df.reset_index(drop=True, inplace=True)
    

    # Create a connection to the SQLite database (or create if it doesn't exist)
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Write the DataFrame to a table (replace 'my_table' with your desired table name)
    df.to_sql(config.RAW_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    logging.info(f"Data successfully written to {config.RAW_TABLE} table.")

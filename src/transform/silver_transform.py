from src.utils.logger import logger

logger.info("Silver transformation started")

import pandas as pd
from pathlib import Path 
pd.set_option('display.max_columns', None)  

BASE_DIR = Path(__file__).resolve().parents[2]

file_path = BASE_DIR / "data" / "raw" / "raw_data_2526.parquet"
silver_data_dir = BASE_DIR / "data" / "silver"


df = pd.read_parquet(file_path)
logger.info("Raw data read successfully")
logger.info(f"Dataframe shape: {df.shape}")

def transform_data(df):
    '''
    Transforms the raw football data by adding new features and performing calculations.
    Args:
        df (pd.DataFrame): The raw football data.
    Returns:
        pd.DataFrame: The transformed football data.
    '''
    logger.info("Starting silver data transformation")

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    df['TotalGoals'] = df['FTHG'] + df['FTAG']

    df['IsHighScoring'] = df['TotalGoals'] > 3

    df['TotalShots'] = df['HS'] + df['AS']

    df['TotalShotsOnTarget'] = df['HST'] + df['AST']

    df['ShotAccuracy'] = df['TotalShotsOnTarget'] / df['TotalShots']

    df['HomePoints'] = df['FTR'].map({'H': 3, 'A': 0, 'D': 1})

    df['AwayPoints'] = df['FTR'].map({'H': 0, 'A': 3, 'D': 1})

    df['BTTS'] = (df['FTHG'] > 0) & (df['FTAG'] > 0)

    df['HomeCleanSheet'] = df['FTAG'] == 0

    df['AwayCleanSheet'] = df['FTHG'] == 0

    logger.info("Silver data transformation completed successfully")
    logger.info(f"Transformed dataframe shape: {df.shape}")
    return df

transformed_df = transform_data(df)
print(transformed_df.head(10))


try:
    silver_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Silver data directory created successfully")

    silver_file_path = silver_data_dir / "transformed_data_2526.parquet"
    transformed_df.to_parquet(silver_file_path, index=False)

    logger.info(f"Transformed data saved successfully to: {silver_file_path}")

except Exception as e:
    logger.error(f"Failed to create silver data directory for parquet file: {e}")
    raise

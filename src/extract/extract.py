from src.utils.logger import logger

logger.info("Extraction started")

import pandas as pd
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

config_path = BASE_DIR / "config" / "config.yaml"
raw_data_dir = BASE_DIR / "data" / "raw"


with open(config_path, "r") as file:
    config = yaml.safe_load(file)

source_url = config["source"]["url"]

def extract_data(source_url):
    '''
    Extracts football data from the github repository source URL.
    Args:
        source_url (str): The URL of the data source.
    Returns:
        pd.DataFrame: The extracted football data.
    '''
    logger.info("Reading data from source")

    if source_url.endswith(".csv"):
        df = pd.read_csv(source_url)
        logger.info("CSV extraction successfully")
        return df
    
    else:
        logger.error("Unsupported file format")
        raise ValueError("Unsupported file format")
    
data = extract_data(source_url)
print(data.head())


try:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Raw data directory created successfully")

    file_path = raw_data_dir / "raw_data_2526.parquet"
    data.to_parquet(file_path, index=False)

    logger.info(f"Data saved successfully to: {file_path}")

except Exception as e:
    logger.error(f"Failed to create raw data directory for parquet file: {e}")
    raise
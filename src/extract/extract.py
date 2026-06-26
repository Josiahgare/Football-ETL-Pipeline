import pandas as pd
import yaml
from pathlib import Path

from src.utils.logger import logger


BASE_DIR = Path(__file__).resolve().parents[2]
config_path = BASE_DIR / "config" / "config.yaml"
raw_data_dir = BASE_DIR / "data" / "raw"

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.error(f"Path not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config



def validate_config(config: dict) -> None:
    """
    Validates required configuration settings for the extraction pipeline.

    Args:
        config (dict): Parsed configuration dictionary.

    Raises:
        KeyError: If required configuration keys are missing.
    """

    if "source" not in config:
        logger.error("'Source' section not found in config file")
        raise KeyError("'Source' section not found in config file")

    if "output" not in config:
        logger.error("'Output' section not found in config file")
        raise KeyError("'Output' section not found in config file")

    if "url" not in config["source"]:
        logger.error("'source.url' not found in config file")
        raise KeyError("'source.url' not found in config file")

    if "raw_file_name" not in config["output"]:
        logger.error("'output.raw_file_name' not found in config file")
        raise KeyError("'output.raw_file_name' not found in config file")
    
    if "silver_file_name" not in config["output"]:
        logger.error("'output.silver_file_name' not found in config file")
        raise KeyError("'output.silver_file_name' not found in config file")
    
    logger.info("Configuration validation passed")



def extract_data(source_url: str) -> pd.DataFrame:
    """
    Retrieves football match data from a CSV source and loads it into a DataFrame.

    The function validates that the provided source URL points to a CSV file
    before reading the data. It serves as the extraction step of the ETL pipeline.

    Args:
        source_url (str): URL or file path of the CSV dataset.

    Returns:
        pd.DataFrame: Extracted football match data.

    Raises:
        ValueError: If the source does not reference a CSV file.
    """

    logger.info("Reading data from source")

    if not source_url.endswith(".csv"):
        logger.error(f"Unsupported file format {source_url}")
        raise ValueError("Source must be a CSV file")
    
    try:
        df = pd.read_csv(source_url)

        if df.empty:
            logger.warning("Extracted dataset is empty")
            raise ValueError("Extracted dataset is empty")
        
        logger.info(f"CSV extraction successfully with {len(df)} match records and {df.shape[1]} columns.")
    
        return df
    
    except Exception:
        logger.exception("Failed to extract data from source")
        raise
        
    


def save_extracted_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Saves extracted raw data to the raw layer in parquet format.

    The function creates the target directory if necessary and persists the
    dataset as a parquet file for downstream processing.

    Args:
        df (pd.DataFrame): Extracted dataset to be saved.
        file_name (str): Name of the output parquet file.

    Raises:
        Exception: If an error occurs while creating directories or writing
            the parquet file.
    """

    try:
        if df.empty:
            logger.warning(f"{file_name} is empty. Saving empty dataset")

        raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw directory ready: {raw_data_dir}")

        raw_file_path = raw_data_dir / file_name
        df.to_parquet(raw_file_path, index=False)

        logger.info(f"Saved {len(df)} rows successfully to: {raw_file_path}")

    except Exception:
        logger.exception(f"Failed to save raw dataset: {file_name}")
        raise


def run_extraction() -> None:
    """
    Executes the extraction workflow for the football ETL pipeline.

    The function retrieves data from the source, stores it in the raw layer,
    and logs the progress of the extraction process.

    Returns:
        None
    """
    logger.info("Extraction started")

    config = load_config(config_path)

    source_url = config["source"]["url"]
    raw_data = extract_data(source_url)

    file_name = config["output"]["raw_file_name"]
    save_extracted_data(raw_data, file_name)

    logger.info(f"Raw data extracted and saved successfully for {len(raw_data)} rows.")


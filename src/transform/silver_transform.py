import pandas as pd
from pathlib import Path 
pd.set_option('display.max_columns', None)  

from src.utils.logger import logger
from src.extract.extract import load_config, config_path


BASE_DIR = Path(__file__).resolve().parents[2]
raw_file_path = BASE_DIR / "data" / "raw" / "raw_data_2526.parquet"
silver_data_dir = BASE_DIR / "data" / "silver"


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Loads the raw football dataset from a parquet file.

    Validates that the source file exists and contains data before returning
    the dataset for downstream transformations.

    Args:
        file_path (Path): Location of the raw parquet dataset.

    Returns:
        pd.DataFrame: Raw football dataset loaded from the specified file.

    Raises:
        FileNotFoundError: If the source file does not exist.
    """

    if not file_path.exists():
        logger.warning(f"{file_path} not found")
        raise 

    df = pd.read_parquet(file_path)

    if df.empty:
        logger.warning("Raw dataset is empty")
    
    logger.info(f"Raw data read succesfully and contains {df.shape[0]} rows and {df.shape[1]} columns.")

    return df
    


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches raw football match data with derived performance and analytics metrics.

    The function converts date fields to datetime format and creates additional
    features including total goals, scoring indicators, shot statistics, match
    points, both-teams-to-score (BTTS) flags, and clean sheet metrics. These
    derived attributes are used in downstream silver and gold layer analyses.

    Args:
        df (pd.DataFrame): Raw football match dataset containing match results
            and performance statistics.

    Returns:
        pd.DataFrame: Enhanced dataset with calculated football analytics
        features ready for further processing.
    """
    logger.info(f"Starting silver data transformation for {len(df)} match records")

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

    logger.info(
    "Created derived features: TotalGoals, IsHighScoring, "
    "ShotAccuracy, HomePoints, AwayPoints, BTTS, CleanSheet metrics")

    logger.info("Silver data transformation completed successfully. "
                f"output shape: {df.shape}")
    return df




def save_silver_transformation(df: pd.DataFrame, file_name: str) -> None:
    """
    Persists the transformed dataset to the silver layer.

    Creates the target directory if it does not already exist and saves the
    DataFrame as a parquet file. Logs the save operation and any errors
    encountered during the process.

    Args:
        df (pd.DataFrame): Transformed dataset to be saved.
        file_name (str): Name of the output parquet file.

    Raises:
        Exception: Propagates any exception encountered while saving the dataset.
    """

    try:
        if df.empty:
            logger.warning(f"{file_name} is empty. Saving empty dataset")

        silver_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Silver data directory created successfully")

        file_path = silver_data_dir / file_name
        df.to_parquet(file_path, index=False)

        logger.info(f"Silver data saved successfully to: {file_path}")

    except Exception as e:
        logger.error(f"Failed to create silver data directory for parquet file: {e}")
        raise


def run_silver_transformation():
    """
    Executes the silver-layer transformation pipeline.

    Loads the raw dataset, applies transformation logic, and saves the
    resulting dataset to the silver layer. Logs the progress and completion
    of the transformation workflow.

    Returns:
        None
    """

    logger.info("Silver transformation started")

    df = load_raw_data(raw_file_path)

    transformed_df = transform_data(df)

    config = load_config(config_path)
    file_name = config["output"]["silver_file_name"]

    save_silver_transformation(transformed_df, file_name)
    logger.info((f"Silver transformation saved successfully for {len(transformed_df)} rows"))


# commits
# define function to load, save, & run silver transformation on raw data as well as include error handling
# delete global execution outside functions

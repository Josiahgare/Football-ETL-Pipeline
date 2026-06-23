import pandas as pd
from pathlib import Path

from src.utils.logger import logger

logger.info("Gold transformation started")

BASE_DIR = Path(__file__).resolve().parents[2]
SILVER_DIR = BASE_DIR / "data" / "silver"
file_path =  SILVER_DIR / "transformed_data_2526.parquet"
gold_data_dir = BASE_DIR / "data" / "gold"

def load_silver_data(file_path: Path):
    """
    Loads the silver-layer dataset from a parquet file.

    This function checks that the specified file exists, reads it into a pandas
    DataFrame, and validates that the dataset is not empty before returning it.
    It ensures early failure if the file is missing or contains no data.

    Args:
        file_path (Path): Path to the silver-layer parquet file.

    Returns:
        pd.DataFrame: Loaded silver dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file exists but contains no records.
    """
    if not file_path.exists():
        logger.error(f"The silver dataset file not found: {file_path}")
        raise FileNotFoundError(f"Silver dataset file not found: {file_path}")

    df = pd.read_parquet(file_path)

    if df.empty:
        logger.error("The silver dataset is empty")
        raise ValueError("Silver dataset is empty")

    logger.info(f"The silver dataset is loaded successfully with {len(df)} rows")
    return df


df = load_silver_data(file_path)

required_columns = []


def validate_column(df, required_columns):
    '''
    Validates that all required columns exist in the input DataFrame before any transformation is applied.

    Args:
        df (pd.DataFrame): The silver-layer football dataset to be validated.
        required_columns (list): List of columns that must be present in the DataFrame for downstream processing.

    Raises:
        ValueError: If one or more required columns are missing from the DataFrame, stopping the pipeline execution.

    Logs:
        - Error log if required columns are missing.
        - Info log when validation passes successfully.
    '''

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(
            f"Missing required columns in silver data: {missing_columns}"
            )
    else:
        logger.info(f"Column validation passed for {len(required_columns)} required columns")



def league_standings(df):
    """
    Generates a league table from match-level football data.

    The function aggregates performance metrics for both home and away matches,
    combines them into a single team-level summary, and ranks teams based on points and goal difference.

    Args:
        df (pd.DataFrame): Cleaned football match dataset containing team
            results and computed points columns.

    Returns:
        pd.DataFrame: Ranked league table showing each team's total matches,
        goals scored, goals conceded, goal difference, and total points.
    """
    logger.info("Starting league standings computation from match data")

    required_columns

    home_standings = df.groupby('HomeTeam').agg(
        Points=('HomePoints', 'sum'),
        MatchesPlayed=('HomeTeam', 'count'),
        GoalsFor=('FTHG', 'sum'),
        GoalsAgainst=('FTAG', 'sum')
    ) 

    away_standings = df.groupby('AwayTeam').agg(
        Points=('AwayPoints', 'sum'),
        MatchesPlayed=('AwayTeam', 'count'),
        GoalsFor=('FTAG', 'sum'),
        GoalsAgainst=('FTHG', 'sum')
    )

    total_standings = home_standings.add(away_standings, fill_value=0)
    total_standings['GoalDifference'] = total_standings['GoalsFor'] - total_standings['GoalsAgainst']
    total_standings = total_standings.sort_values(by=['Points', 'GoalDifference'], ascending=False).reset_index()
    total_standings.index = range(1, len(total_standings) + 1)

    total_standings = total_standings[['HomeTeam', 'MatchesPlayed', 'GoalsFor', 'GoalsAgainst', 'GoalDifference', 'Points']].rename(columns={'HomeTeam': 'Team'})

    logger.info("League standings generated successfully")
    return total_standings


league_table = league_standings(df)


def home_performance(df):
    '''
    Calculates the performance of a team based on its home matches.
    Args:
        df (pd.DataFrame): The transformed football data.
    Returns:
        pd.DataFrame: The home performance metrics including wins, draws, losses, and clean sheets.
    '''
    logger.info("Calculating home performance metrics")

    home_perf_df = df.groupby('HomeTeam').agg(
        Wins=('HomePoints', lambda x: (x == 3).sum()),
        Draws=('HomePoints', lambda x: (x == 1).sum()),
        Losses=('HomePoints', lambda x: (x == 0).sum()),
        CleanSheets=('HomeCleanSheet', 'sum'),
        Corners=('HC', 'sum'),
        GoalsScored=('FTHG', 'sum'),
        YellowCards=('HY', 'sum'),
        RedCards=('HR', 'sum')
    )
    
    home_perf_df['HomeTeam'] = home_perf_df.index
    home_perf_df = home_perf_df.sort_values(by=['Wins', 'Draws'], ascending=False).reset_index(drop=True)
    home_perf_df.index = range(1, len(home_perf_df) + 1)
    home_perf_df['Points'] = (home_perf_df['Wins'] * 3) + (home_perf_df['Draws'] * 1)
    home_perf_df = home_perf_df[['HomeTeam', 'Points', 'Wins', 'Draws', 'Losses', 'CleanSheets', 'Corners', 'GoalsScored', 'YellowCards', 'RedCards']]

    logger.info("Home performance metrics calculated succesfully")
    return home_perf_df


league_home_performance = home_performance(df)


def away_performance(df):
    '''
    Calculates the performance of a team based on its away matches.
    Args:
        df (pd.DataFrame): The transformed football data.
    Returns:
        pd.DataFrame: The away performance metrics including wins, draws, losses, and clean sheets.
    '''
    logger.info("Calculating away performance metrics")

    away_perf_df = df.groupby('AwayTeam').agg(
        Wins=('AwayPoints', lambda x: (x == 3).sum()),
        Draws=('AwayPoints', lambda x: (x == 1).sum()),
        Losses=('AwayPoints', lambda x: (x == 0).sum()),
        CleanSheets=('AwayCleanSheet', 'sum'),
        Corners=('AC', 'sum'),
        GoalsScored=('FTAG', 'sum'),
        YellowCards=('AY', 'sum'),
        RedCards=('AR', 'sum')
    )
    
    away_perf_df['AwayTeam'] = away_perf_df.index
    away_perf_df = away_perf_df.sort_values(by=['Wins', 'Draws'], ascending=False).reset_index(drop=True)
    away_perf_df.index = range(1, len(away_perf_df) + 1)
    away_perf_df['Points'] = (away_perf_df['Wins'] * 3) + (away_perf_df['Draws'] * 1)
    away_perf_df = away_perf_df[['AwayTeam', 'Points', 'Wins', 'Draws', 'Losses', 'CleanSheets', 'Corners', 'GoalsScored', 'YellowCards', 'RedCards']]

    logger.info("Away performance metrics calculated succesfully")
    return away_perf_df

league_away_performance = away_performance(df)



def monthly_performance(df, month):
    '''
    Calculate the performance of a team on a monthly bases.
    Args:
        df (pd.DataFrame): The transformed football data.
        month: The numeric value of the month interested.
    Returns:
        pd.DataFrame: The performance of the team in a given month.
    '''
    logger.info("Calculating monthly performance metric")

    month_df = df[df['Date'].dt.month == month]

    hp = home_performance(month_df).set_index('HomeTeam')
    ap = away_performance(month_df).set_index('AwayTeam')
    
    monthly_perf_df = hp.add(ap, fill_value=0)
    monthly_perf_df = monthly_perf_df.sort_values(by=['Wins', 'Draws'], ascending=False).reset_index()
    monthly_perf_df['Points'] = (monthly_perf_df['Wins'] * 3) + (monthly_perf_df['Draws'] * 1)
    monthly_perf_df = monthly_perf_df[['HomeTeam', 'Wins', 'Draws', 'Losses', 'Points']].rename(columns={'HomeTeam': 'Team'})
    monthly_perf_df.index = range(1, len(monthly_perf_df) + 1)

    logger.info("Monthly performance metrics calculated succesfully")
    return monthly_perf_df

august_performance = monthly_performance(df, 8)

# Task 1: merge home performance and away performance 
# Task 2: input required columns and call validate_column() function


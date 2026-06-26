import pandas as pd
from pathlib import Path

from src.utils.logger import logger


BASE_DIR = Path(__file__).resolve().parents[2]
SILVER_DIR = BASE_DIR / "data" / "silver"
file_path =  SILVER_DIR / "transformed_data_2526.parquet"
gold_data_dir = BASE_DIR / "data" / "gold"

def load_silver_data(file_path: Path) -> pd.DataFrame:
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




def validate_column(df: pd.DataFrame, required_columns: list) -> None:
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



def league_standings(df: pd.DataFrame) -> pd.DataFrame:
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

    required_columns = ['HomeTeam', 'AwayTeam', 'HomePoints', 'AwayPoints', 'FTHG','FTAG']
    validate_column(df, required_columns)

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



def home_away_performance(df:pd.DataFrame, venue: str= "home") -> pd.DataFrame:
    """
    Computes team performance metrics for either home or away matches.

    The function dynamically selects relevant match columns based on the
    specified venue and aggregates team-level statistics such as wins,
    draws, losses, goals scored, defensive records, and disciplinary metrics.
    The output is sorted by overall performance ranking.

    Args:
        df (pd.DataFrame): Cleaned football match dataset containing match
            outcomes and performance-related fields.
        venue (str): Match context to analyze. Must be either 'home' or 'away'.

    Returns:
        pd.DataFrame: Aggregated team performance table sorted by points,
        wins, and draws.
    """

    logger.info(f"Starting {venue} performance computation")

    if venue not in ["home", "away"]:
        logger.error(f"Invalid venue provided: {venue}")
        raise ValueError("venue must be either 'home' or 'away'")
    
    config = {
        "home": {
            "team": "HomeTeam",
            "points": "HomePoints",
            "clean_sheet": "HomeCleanSheet",
            "corners": "HC",
            "goals_scored": "FTHG",
            "yellow_cards": "HY",
            "red_cards": "HR"
        },
        "away": {
            "team": "AwayTeam",
            "points": "AwayPoints",
            "clean_sheet": "AwayCleanSheet",
            "corners": "AC",
            "goals_scored": "FTAG",
            "yellow_cards": "AY",
            "red_cards": "AR"
        }
    }

    cols = config[venue]

    required_columns = list(cols.values())
    validate_column(df, required_columns)

    logger.info(f"Aggregating {venue} stats by {cols['team']}")

    perf_df = df.groupby(cols['team']).agg(
        Wins=(cols['points'], lambda x: (x == 3).sum()),
        Draws=(cols['points'], lambda x: (x == 1).sum()),
        Losses=(cols['points'], lambda x: (x == 0).sum()),
        CleanSheets=(cols['clean_sheet'], 'sum'),
        Corners=(cols['corners'], 'sum'),
        GoalsScored=(cols['goals_scored'], 'sum'),
        YellowCards=(cols['yellow_cards'], 'sum'),
        RedCards=(cols['red_cards'], 'sum')
    )
    
    perf_df['Points'] = (perf_df['Wins'] * 3) + (perf_df['Draws'] * 1)
    perf_df = perf_df.sort_values(by=['Points', 'Wins', 'Draws'], ascending=False).reset_index()
    perf_df = perf_df[[cols['team'], 'Points', 'Wins', 'Draws', 'Losses', 'CleanSheets', 'Corners', 'GoalsScored', 'YellowCards', 'RedCards']]

    logger.info(f"{venue.capitalize()} performance computation completed successfully")
    return perf_df



def monthly_performance(df: pd.DataFrame, month: int) -> pd.DataFrame:
    '''
    Calculate the performance of a team on a monthly bases.
    Args:
        df (pd.DataFrame): The transformed football data.
        month: The numeric value of the month interested.
    Returns:
        pd.DataFrame: The performance of the team in a given month.
    '''
    logger.info(f"Calculating monthly performance metric for {len(df)} rows")

    if month not in range(1, 13):
        raise ValueError("Month value must be between 1 and 12")

    required_columns = ['Date']
    validate_column(df, required_columns)

    month_df = df[df['Date'].dt.month == month]

    if month_df.empty:
        logger.warning(f"No matches found for month {month}")
        return pd.DataFrame()

    hp_df = home_away_performance(month_df, 'home').set_index('HomeTeam')
    ap_df = home_away_performance(month_df, 'away').set_index('AwayTeam')
    
    monthly_perf_df = hp_df.add(ap_df, fill_value=0)
    monthly_perf_df['Points'] = (monthly_perf_df['Wins'] * 3) + (monthly_perf_df['Draws'] * 1)
    monthly_perf_df = monthly_perf_df.sort_values(by=['Points', 'Wins', 'Draws'], ascending=False).reset_index()
    monthly_perf_df = monthly_perf_df[['HomeTeam', 'Wins', 'Draws', 'Losses', 'Points']].rename(columns={'HomeTeam': 'Team'})
    monthly_perf_df.index = range(1, len(monthly_perf_df) + 1)

    logger.info(f"Monthly performance metrics calculated successfully for {len(monthly_perf_df)} teams")
    return monthly_perf_df



def save_gold_transformation(df, file_name):

    try:
        if df.empty:
            logger.warning(f"{file_name} is empty. Saving empty dataset")

        gold_data_dir.mkdir(parents=True, exist_ok=True)

        file_path = gold_data_dir / file_name

        df.to_csv(file_path, index=False)

        logger.info(f"Saved {file_name} successfully for {len(df)} rows")

    except Exception:
        logger.exception(f"Failure to save gold output for {file_name}")
        raise



def run_gold_transformation():
    logger.info("Gold transformation started")

    df = load_silver_data(file_path)

    league_table = league_standings(df)
    home_stats = home_away_performance(df, "home")
    away_stats = home_away_performance(df, 'away')
    august_month_stats = monthly_performance(df, 8)

    gold_outputs = {
        'league_table.csv': league_table,
        'home_stats.csv': home_stats,
        'away_stats.csv': away_stats,
        'august_month_stats.csv': august_month_stats
    }

    for file_name, output_df in gold_outputs.items():
        save_gold_transformation(output_df, file_name)
    
    logger.info(f"Gold transformation saved successfully. "
                f"{len(output_df)} gold outputs generated")

if __name__ == "__main__":
    run_gold_transformation()


# commits
# defined function to save & run gold transformations, also define a function to replace two similar functions
# deleted two similar functions and global execution outside defined functions

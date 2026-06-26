import time
from src.utils.logger import logger
from src.extract.extract import run_extraction
from src.transform.silver_transform import run_silver_transformation
from src.transform.gold_transform import run_gold_transformation

def run_pipeline()-> None:
    run_start_time = time.time()
    
    try:
        logger.info("Starting ETL pipeline run")
        logger.info("=" * 30)

        run_extraction()

        run_silver_transformation()

        run_gold_transformation()

        logger.info("Pipeline run completed")

    except Exception:
        logger.exception("Pipeline execution failed")
        raise

    finally:
        run_end_time = round(time.time() - run_start_time, 2)
        logger.info(f"Pipeline execution time: {run_end_time} seconds")


if __name__ == "__main__":
    run_pipeline()


# commit
# create a pipeline script to run entire pipeline
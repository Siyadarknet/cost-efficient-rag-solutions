from pathlib import Path
from loguru import logger
from app.config import settings

# Create log directory
Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)

# Remove default logger
logger.remove()

# Console logging
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "{message}"
)

# General application log
logger.add(
    f"{settings.LOG_DIR}/app.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO",
    format="{time} | {level} | {message}"
)

# Error log
logger.add(
    f"{settings.LOG_DIR}/error.log",
    rotation="5 MB",
    retention="20 days",
    level="ERROR",
    backtrace=True,
    diagnose=True
)

# Query log (JSON)
logger.add(
    f"{settings.LOG_DIR}/queries.jsonl",
    serialize=True,
    level="INFO"
)


def log_query(
    question: str,
    latency: float,
    retrieved_chunks: int,
    prompt_tokens: int,
    completion_tokens: int,
):
    """
    Log every query with important metrics.
    """

    logger.info(
        {
            "question": question,
            "latency_seconds": round(latency, 3),
            "retrieved_chunks": retrieved_chunks,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    )
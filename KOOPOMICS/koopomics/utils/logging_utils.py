import logging
import os

def setup_run_logger(logs_dir: str, name: str = "koopomics", level=logging.INFO) -> logging.Logger:
    """
    Configure a logger that writes both to console and to a run-specific log file,
    showing class/module context for each message.
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 🛑 Prevent duplicate handlers
    if logger.handlers:
        return logger

    # === Console formatter ===
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s | %(message)s",
        "%H:%M:%S",
    )

    # === File formatter ===
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # === Handlers ===
    console = logging.StreamHandler()
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 🚫 Prevent log propagation to root (avoids duplicate "INFO:koopomics:" lines)
    logger.propagate = False

    logger.info(f"🪵 Logging initialized → {log_path}")
    return logger

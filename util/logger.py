import logging
import logging.config
from typing import Optional

from util.tool_function import safe_dir

logger = logging.getLogger(__name__)

def setup_logging(log_path: Optional[str] = None, level: str = "DEBUG"):
    handlers_dict = {
        "console_handler":{
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    }
    if log_path is not None:
        log_path = safe_dir(log_path, with_filename=True)
        handlers_dict["file_handler"] = {
            "class": "logging.FileHandler",
            "formatter": "full",
            "level": "DEBUG",
            "filename": log_path,
            "encoding": "utf8"
        }
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters":{
            "simple":{
                "format": "[ %(asctime)s ] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "full":{
                "format": "[ %(asctime)s ] %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": handlers_dict,
        "loggers":{
            "experiment":{
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "__main__":{
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "models": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "training": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "util": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "evaluate": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
        }
    }
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger().handlers[0].setLevel(logging.WARNING)

    logging.config.dictConfig(config_dict)
    logger.info("Setup Logging!")

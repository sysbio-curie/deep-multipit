import logging
import logging.config
from pathlib import Path
from dmultipit.utils import read_json
import os


def setup_logging(save_dir,  default_level=logging.INFO):
    """
    Setup logging configuration
    """

    # Fix error to access logger_config.json
    log_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logger_config.json')
    log_config = Path(log_config)

    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)

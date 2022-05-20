import logging
from logging import Formatter, StreamHandler, DEBUG, INFO, WARNING

def set_up_logging(console_level):
    log_console_format = "[%(levelname)s] - %(name)s -> %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(DEBUG)

    console_handler = StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(Formatter(log_console_format))
    root_logger.addHandler(console_handler)

import logging

import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    light_green = "\x1b[92;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    white = "\x1b[37;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    format = "[%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: light_green + format + reset,
        logging.INFO: format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Create a filter to exclude logs from external libraries
class ExcludeSpecificLogsFilter(logging.Filter):
    def __init__(self, excluded_strings):
        self.excluded_strings = excluded_strings

    def filter(self, record):
        for string in self.excluded_strings:
            try:
                if record.msg and str(record.msg).startswith(string):
                    return False
            except Exception as e:
                print("Logger error:", e)
                return False
        return True


excluded_strings = [
    "DEBUG:docker",
    "DEBUG:matplotlib",
    "DEBUG:urllib3",
    "DEBUG:trimesh",
]

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler("bbn.log", mode="w")
file_handler.setLevel(logging.DEBUG)
# Apply the custom filter to exclude specific log messages
file_handler.addFilter(ExcludeSpecificLogsFilter(excluded_strings))
# Create a formatter and set it for the file handler
formatter = CustomFormatter()
file_handler.setFormatter(formatter)

# Create a console handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Apply the custom filter to exclude specific log messages in the console
console_handler.addFilter(ExcludeSpecificLogsFilter(excluded_strings))
# Set the formatter for the console handler
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

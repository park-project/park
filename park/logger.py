import logging


level = logging.INFO
logging.basicConfig(level=level)


def debug(msg, *args, **kwargs):
    logging.debug(msg, args, kwargs)


def info(msg, *args, **kwargs):
    logging.info(msg, args, kwargs)


def warn(msg, *args, **kwargs):
    logging.warning(msg, args, kwargs)


def error(msg, *args, **kwargs):
    logging.error(msg, args, kwargs)

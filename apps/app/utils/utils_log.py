import pathlib
import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


class LogFactory(object):
    @classmethod
    def add_handler(cls, input_logger, input_handler, input_format, input_level):
        input_handler.setFormatter(input_format)
        input_handler.setLevel(input_level)
        input_logger.addHandler(input_handler)
        return input_logger

    @classmethod
    def get_log(cls, log_filename, max_bytes=10 * 1024 * 1024, backup_count=5):
        log_level = 20
        log_filepath = '/var/log/apps/{}.log'.format(log_filename)
        logger = logging.getLogger(log_filename)
        if not logger.handlers:
            pathlib.Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(log_filepath, mode='a', maxBytes=max_bytes,
                                               backupCount=backup_count, encoding='utf-8')
            log_format = logging.Formatter(
                '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            logger = cls.add_handler(logger, file_handler, log_format, log_level)
        logger.setLevel(log_level)
        return logger

    @classmethod
    def get_stream_log(cls):
        """
        :param log_filename: The name of log file
        :param log_format_setting: The format of log
        :param log_level: The level of log
        :return: The instance of stream logger
        """
        log_level = 20
        logger = logging.getLogger('audit.log')
        if not logger.handlers:
            file_handler = StreamHandler()
            log_format = logging.Formatter(
                '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            logger = cls.add_handler(logger, file_handler, log_format, log_level)
        logger.setLevel(log_level)
        return logger

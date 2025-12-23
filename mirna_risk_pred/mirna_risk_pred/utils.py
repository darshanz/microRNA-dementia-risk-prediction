import configparser
import os
import time
import logging

def create_logger(out_dir):
    log_dir = f"{out_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join('logs',
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # initialize console
    #console = logging.StreamHandler()
    #console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    #logger.addHandler(console)

    return logger


def read_config_file(config_file):  
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    cf = configparser.ConfigParser()
    cf.read(config_file)
    cfg = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return cfg
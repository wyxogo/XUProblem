import logging
import argparse

import os
import sys

class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val
        self.cnt += n
        self.avg = self.sum / self.cnt

def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

def get_arguments():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('VGG19_10')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--mini_label_names', type=list, default=['maple', 'bed', 'bus', 'plain', 'dolphin',
                                                                    'bottle', 'cloud', 'bridge', 'baby', 'rocket'])
    # parser.add_argument('--best_model_path', type=str, default='./outputs/best_model/')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=int, default=1e-4)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_model_step', type=int, default=25)
    parser.add_argument('--test_steps', type=int, default=10)
    
    parser.add_argument('--mean', type=list, default=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
    parser.add_argument('--std', type=int, default=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])    
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--numclass', type=int, default=10)
    parser.add_argument('--output', type=str, default='./outputs/')

    arguments = parser.parse_args()
    return arguments

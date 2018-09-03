import numpy as np
import tensorflow as tf
import config
from config import get_config
from utils import prepare_dirs_and_logger, save_config


def main(manual):
    prepare_dirs_and_logger(manual)
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)
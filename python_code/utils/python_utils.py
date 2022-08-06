import pickle as pkl
from typing import Dict, Any

import numpy as np

from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config

conf = Config()


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str) -> Dict[Any, Any]:
    output = open(pkls_path, 'rb')
    return pkl.load(output)


def normalize_for_modulation(size: int) -> int:
    """
    Return size if BPSK, or 0.5 * size if QPSK. This is the amount of symbols in tx/rx words
    """
    return size * 2 // MODULATION_NUM_MAPPING[conf.modulation_type]

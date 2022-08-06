import pickle as pkl
from typing import Dict, Any

import numpy as np

from python_code.utils.config_singleton import Config

conf = Config()


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str) -> Dict[Any, Any]:
    output = open(pkls_path, 'rb')
    return pkl.load(output)

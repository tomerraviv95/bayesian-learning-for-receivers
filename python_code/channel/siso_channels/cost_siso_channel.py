import os

import numpy as np
import scipy.io
from numpy.random import default_rng

from dir_definitions import SISO_COST2100_DIR
from python_code.utils.config_singleton import Config

conf = Config()

COST_LENGTH = 200
COST_STEP = 2


class Cost2100SISOChannel:
    @staticmethod
    def calculate_channel(memory_length: int, fading: bool = False, index: int = 0) -> np.ndarray:
        total_h = np.empty([COST_LENGTH // COST_STEP, memory_length])
        for i in range(memory_length):
            h_channel_response = scipy.io.loadmat(os.path.join(SISO_COST2100_DIR, f'h_{i}'))
            total_h[:, i] = h_channel_response['h_channel_response_mag'].reshape(-1)[:COST_LENGTH][::COST_STEP]
        h = np.reshape(total_h[index], [1, memory_length])
        h *= 0.8
        return h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, memory_length: int) -> np.ndarray:
        """
        The SISO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :param memory_length: length of channel memory
        :return: received word
        """
        conv = Cost2100SISOChannel._compute_channel_signal_convolution(h, memory_length, s)
        [row, col] = conv.shape
        w = Cost2100SISOChannel._sample_noise_vector(row, col, snr)
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, memory_length: int, s: np.ndarray) -> np.ndarray:
        blockwise_s = np.concatenate([s[:, i:-memory_length + i] for i in range(memory_length)], axis=0)
        conv = np.dot(h[:, ::-1], blockwise_s)
        return conv

    @staticmethod
    def _sample_noise_vector(row: int, col: int, snr: float) -> np.ndarray:
        noise_generator = default_rng(seed=conf.seed)
        snr_value = 10 ** (snr / 10)
        w = (snr_value ** (-0.5)) * noise_generator.standard_normal((row, col))
        return w

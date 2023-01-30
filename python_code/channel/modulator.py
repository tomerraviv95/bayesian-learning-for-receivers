import math

import numpy as np
import torch

from python_code import DEVICE
from python_code.utils.constants import HALF


class BPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        x = 1 - 2 * c
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """
        return HALF * (s + 1)


class QPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        QPSK modulation
        [0,0] -> [1/sqrt(2),1/sqrt(2)]
        [0,1] -> [1/sqrt(2),-1/sqrt(2)]
        [1,0] -> [-1/sqrt(2),1/sqrt(2)]
        [1,1] -> [-1/sqrt(2),-1/sqrt(2)]
        :param c: the binary codeword
        :return: modulated signal
        """
        x = (-1) ** c[:, ::2] / np.sqrt(2) + (-1) ** c[:, 1::2] / np.sqrt(2) * 1j
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        return ((-1) * HALF * (torch.view_as_real(s) - 1)).transpose(1, 2).reshape(-1, s.shape[1])


class EightPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        8PSK modulation
        [0,0,0] -> [-1,0]
        [0,0,1] -> [-1/sqrt(2),-1/sqrt(2)]
        [0,1,0] -> [-1/sqrt(2),1/sqrt(2)]
        [0,1,1] -> [0,-1]
        [1,0,0] -> [0,1]
        [1,0,1] -> [1/sqrt(2),-1/sqrt(2)]
        [1,1,0] -> [1/sqrt(2),1/sqrt(2)]
        [1,1,1] -> [1,0]
        :param c: the binary codeword
        :return: modulated signal
        """
        deg = (c[:, ::3] / 4 + c[:, 1::3] / 2 + c[:, 2::3]) * math.pi
        x = np.exp(1j * deg)
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(s.imag, s.real) / np.pi
        theta[theta < 0] += 2
        c1 = torch.div(theta, 0.25, rounding_mode='floor') % 2
        c2 = torch.div(theta, 0.5, rounding_mode='floor') % 2
        c3 = torch.div(theta, 1, rounding_mode='floor') % 2
        concat_cs = torch.zeros(3 * s.shape[0], s.shape[1]).to(DEVICE)
        concat_cs[::3, :] = c1
        concat_cs[1::3, :] = c2
        concat_cs[2::3, :] = c3
        return concat_cs


MODULATION_DICT = {
    'BPSK': BPSKModulator,
    'QPSK': QPSKModulator,
    'EightPSK': EightPSKModulator
}

MODULATION_NUM_MAPPING = {
    'BPSK': 2,
    'QPSK': 4,
    'EightPSK': 8
}

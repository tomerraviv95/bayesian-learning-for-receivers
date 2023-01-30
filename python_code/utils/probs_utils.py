import itertools

import numpy as np
import torch

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.constants import HALF, ModulationType



def generate_bits_by_state(state: int, n_state: int) -> torch.Tensor:
    """
    Calculates all possible combinations of vector of length state, with elements from 0,..,n_states-1
    Given by the constellation size.
    """
    combinations = list(itertools.product(range(MODULATION_NUM_MAPPING[conf.modulation_type]), repeat=n_state))
    return torch.Tensor(combinations[state][::-1]).reshape(1, n_state).to(DEVICE)


def calculate_mimo_states(n_user: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    Calculates mimo states vector for the transmitted words. Number of states is 2/4/8 ** n_user depending on the
    constellation size.
    """
    states_enumerator = (MODULATION_NUM_MAPPING[conf.modulation_type] ** torch.arange(n_user)).to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def calculate_symbols_from_states(state_size: int, gt_states: torch.Tensor) -> torch.Tensor:
    """
    Used for the dnn-aided receivers. Calculates the symbols from the states.
    """
    mask = MODULATION_NUM_MAPPING[conf.modulation_type] ** torch.arange(state_size).to(DEVICE, gt_states.dtype)
    if conf.modulation_type == ModulationType.BPSK.name:
        return gt_states.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    elif conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]:
        result = torch.div(gt_states.unsqueeze(-1), mask, rounding_mode='floor')
        result = result % MODULATION_NUM_MAPPING[conf.modulation_type]
        return result


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    """
    return torch.sign(p - HALF)


def prob_to_QPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to QPSK Symbols by hard threshold.
    first bit: [0,0.5] -> '+1',[0.5,1] -> '-1'
    second bit: [0,0.5] -> '+1',[0.5,1] -> '-1'
    """
    p_real_neg = p[:, :, 0] + p[:, :, 2]
    first_symbol = (-1) * torch.sign(p_real_neg - HALF)
    p_img_neg = p[:, :, 1] + p[:, :, 2]
    second_symbol = (-1) * torch.sign(p_img_neg - HALF)
    s = torch.cat([first_symbol.unsqueeze(-1), second_symbol.unsqueeze(-1)], dim=-1)
    return torch.view_as_complex(s)


def prob_to_EightPSK_symbol(probs_vec: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to 8PSK Symbols by hard threshold.
    000 -> 1
    001 -> 1 / sqrt(2) + j / sqrt(2)
    010 -> j
    011 -> - 1 / sqrt(2) + j / sqrt(2)
    100 -> -1
    101 -> -1 / sqrt(2) - j / sqrt(2)
    110 -> -j
    111 -> 1 / sqrt(2) - j / sqrt(2)
    Note that prob of 0 is missing as it is 1-sum(probs[1:7]), so probs_vec start from p_001
    Sums the odd probabilities (1,3,5,7) to get the estimate for the right bit
    Sums the probabilities (2,3,6,7) to get the estimate for the middle bit
    Sums the probabilities (4,5,6,7) to get the estimate for the left bit
    """
    p = probs_vec
    s1_p = p[:, :, 0] + p[:, :, 2] + p[:, :, 4] + p[:, :, 6]
    first_bit = torch.div(torch.sign(s1_p - HALF) + 1, 2, rounding_mode='floor')
    s2_p = p[:, :, 1] + p[:, :, 2] + p[:, :, 5] + p[:, :, 6]
    second_bit = torch.div(torch.sign(s2_p - HALF) + 1, 2, rounding_mode='floor')
    s3_p = p[:, :, 3] + p[:, :, 4] + p[:, :, 5] + p[:, :, 6]
    third_bit = torch.div(torch.sign(s3_p - HALF) + 1, 2, rounding_mode='floor')
    deg = first_bit * np.pi / 4 + second_bit * np.pi / 2 + third_bit * np.pi
    s = torch.exp(1j * deg)
    return s


def get_qpsk_symbols_from_bits(b: np.ndarray) -> np.ndarray:
    return b[::2] + 2 * b[1::2]


def get_eightpsk_symbols_from_bits(b: np.ndarray) -> np.ndarray:
    return b[::3] + 2 * b[1::3] + 4 * b[2::3]


def get_bits_from_qpsk_symbols(target: torch.Tensor) -> torch.Tensor:
    first_bit = target % 2
    second_bit = torch.floor(target / 2) % 2
    target = torch.cat([first_bit.unsqueeze(-1), second_bit.unsqueeze(-1)], dim=2).transpose(1, 2).reshape(
        2 * first_bit.shape[0], -1)
    return target


def get_bits_from_eightpsk_symbols(target: torch.Tensor) -> torch.Tensor:
    first_bit = target % 2
    second_bit = torch.floor(target / 2) % 2
    third_bit = torch.floor(target / 4) % 2
    target = torch.cat([first_bit.unsqueeze(-1), second_bit.unsqueeze(-1), third_bit.unsqueeze(-1)], dim=2)
    target = target.transpose(1, 2).reshape(3 * first_bit.shape[0], -1)
    return target

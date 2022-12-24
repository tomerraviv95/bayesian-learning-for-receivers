from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from python_code.utils.trellis_utils import create_transition_table, acs_block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class VADetector(nn.Module):
    """
    This module implements the classic VA detector
    """

    def __init__(self, n_states: int, memory_length: int):

        super(VADetector, self).__init__()
        self.memory_length = memory_length
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)
        self.softmax = torch.nn.Softmax(dim=1)

    def compute_state_priors(self, h: torch.Tensor) -> torch.Tensor:
        all_states_decimal = np.arange(self.n_states).astype(np.uint8).reshape(-1, 1)
        all_states_binary = np.unpackbits(all_states_decimal, axis=1).astype(int)
        all_states_symbols = BPSKModulator.modulate(all_states_binary[:, -self.memory_length:])
        state_priors = np.dot(all_states_symbols, h.cpu().numpy().T)
        return torch.Tensor(state_priors).to(device)

    def compute_likelihood_priors(self, y: torch.Tensor, h: torch.Tensor):
        # compute priors
        state_priors = self.compute_state_priors(h)
        priors = y - state_priors.T.repeat(repeats=[y.shape[0] // state_priors.shape[1], 1])
        # to llr representation
        snr_value = 10 ** (conf.snr / 10)
        sigma = (snr_value ** (-HALF))
        priors2 = torch.exp(- priors ** 2 / (2 * sigma ** 2))
        return priors2 / priors2.sum(dim=1).reshape(-1, 1)

    def forward(self, rx: torch.Tensor, phase: Phase, h: torch.Tensor) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(device)

        # compute transition likelihood priors
        probs = self.compute_likelihood_priors(rx, h)
        priors = torch.log(probs)

        if phase == Phase.TEST:
            confident_bits = (torch.argmax(probs, dim=1) % 2).reshape(-1, 1)
            confidence_word = torch.amax(probs, dim=1).reshape(-1, 1)
            detected_word = torch.zeros(rx.shape).to(DEVICE)
            for i in range(rx.shape[0]):
                # get the lsb of the state
                detected_word[i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return detected_word, (confident_bits, confidence_word)
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")

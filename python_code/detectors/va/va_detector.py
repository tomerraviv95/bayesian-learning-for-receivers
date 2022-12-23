import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.channel.modulator import BPSKModulator
from python_code.utils.constants import Phase
from python_code.utils.trellis_utils import create_transition_table, acs_block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def compute_state_priors(self, h: torch.Tensor) -> torch.Tensor:
        all_states_decimal = np.arange(self.n_states).astype(np.uint8).reshape(-1, 1)
        all_states_binary = np.unpackbits(all_states_decimal, axis=1).astype(int)
        all_states_symbols = BPSKModulator.modulate(all_states_binary[:, -self.memory_length:])
        state_priors = np.dot(all_states_symbols, h.cpu().numpy().T)
        return torch.Tensor(state_priors).to(device)

    def compute_likelihood_priors(self, y: torch.Tensor, h: torch.Tensor):
        # compute priors
        state_priors = self.compute_state_priors(h)
        priors = y.unsqueeze(dim=2) - state_priors.T.repeat(
            repeats=[y.shape[0] // state_priors.shape[1], 1]).unsqueeze(
            dim=1)
        # to llr representation
        priors = priors ** 2 / 2 - math.log(math.sqrt(2 * math.pi))
        return -priors.reshape(y.shape[0], -1)

    def forward(self, rx: torch.Tensor, phase: Phase, h: torch.Tensor) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(device)

        # compute transition likelihood priors
        priors = self.compute_likelihood_priors(rx, h)

        if phase == Phase.TEST:
            confident_bits = (torch.argmax(torch.exp(priors), dim=1) % 2).reshape(-1, 1)
            confidence_word = torch.amax(torch.exp(priors), dim=1).reshape(-1, 1)
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

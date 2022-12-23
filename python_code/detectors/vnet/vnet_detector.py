from typing import Tuple

import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.utils.constants import Phase
from python_code.utils.trellis_utils import create_transition_table, acs_block

HIDDEN1_SIZE = 100


class VNETDetector(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int):

        super(VNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self._initialize_dnn()
        # self.T = 2  # nn.Parameter(torch.ones(1))

    def _initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN1_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: Phase, index: int = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        if phase == Phase.TRAIN:
            func = torch.nn.LogSoftmax(dim=1)
            priors = func(self.net(rx))
        else:
            func = torch.nn.Softmax(dim=1)
            probs = func(self.net(rx).clone().detach())
            confident_bits = (torch.argmax(probs, dim=1) % 2).reshape(-1, 1)
            confidence_word = torch.amax(probs, dim=1).reshape(-1, 1)
            priors = torch.log(probs)

        if phase == Phase.TEST:
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
            return priors

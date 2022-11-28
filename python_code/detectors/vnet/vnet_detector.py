import numpy as np
import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.utils.constants import Phase

HIDDEN1_SIZE = 75


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    previous state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


def acs_block(in_prob: torch.Tensor, llrs: torch.Tensor, transition_table: torch.Tensor, n_states: int) -> [
    torch.Tensor, torch.LongTensor]:
    """
    Viterbi ACS block
    :param in_prob: last stage probabilities, [batch_size,n_states]
    :param llrs: edge probabilities, [batch_size,1]
    :param transition_table: transitions
    :param n_states: number of states
    :return: current stage probabilities, [batch_size,n_states]
    """
    transition_ind = transition_table.reshape(-1).repeat(in_prob.size(0)).long()
    batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * n_states)
    trellis = (in_prob + llrs)[batches_ind, transition_ind]
    reshaped_trellis = trellis.reshape(-1, n_states, 2)
    return torch.min(reshaped_trellis, dim=2)[0]


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

    def _initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN1_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: Phase) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: Phase.TRAIN or Phase.TEST
        :returns if in Phase.TRAIN - the estimated priors [batch_size,transmission_length,n_states]
        if in Phase.TEST - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        func = torch.nn.LogSoftmax(dim=1)
        priors = func(self.net(rx))

        if phase == Phase.TEST:
            detected_word = torch.zeros(rx.shape).to(DEVICE)
            confidence_word = torch.zeros(rx.shape).to(DEVICE)
            for i in range(rx.shape[0]):
                # get the lsb of the state
                detected_word[i] = torch.argmin(in_prob, dim=1) % 2
                confidence_word[i] = torch.amax(torch.softmax(-in_prob, dim=1), dim=1)
                # run one Viterbi stage
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return detected_word, 2 * confidence_word
        else:
            return priors

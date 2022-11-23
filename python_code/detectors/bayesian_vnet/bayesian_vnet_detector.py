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


class BayesianDNN(nn.Module):
    def __init__(self, n_states, length_scale):
        super(BayesianDNN, self).__init__()
        self.fc1 = nn.Linear(1, HIDDEN1_SIZE).to(DEVICE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, n_states).to(DEVICE)
        self.dropout_logit = nn.Parameter(torch.rand(HIDDEN1_SIZE).reshape(1, -1))
        self.activ = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid()
        self.length_scale = length_scale

    def forward(self, raw_input, num_ensemble, phase):
        prob = 0
        if phase == Phase.TRAIN:
            ARM_ori = []
            ARM_tilde = []
            u_list = []

        for ind_ensemble in range(num_ensemble):

            # first layer
            x = self.activ(self.fc1(raw_input))
            u1 = torch.rand(x.shape).to(DEVICE)
            x = self.dropout_ori(x, self.dropout_logit, u1)
            if phase == Phase.TRAIN:
                x_tilde = self.activ(self.fc1(raw_input))
                x_tilde = self.dropout_tilde(x_tilde, self.dropout_logit, u1)
                u_list.append(u1)
            else:
                pass

            # second layer
            x = self.fc2(x)
            prob += x
            if phase == Phase.TRAIN:
                ARM_ori.append(x)
                ARM_tilde.append(x_tilde)

        ## KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.length_scale ** 2 / 2) * torch.sigmoid(self.dropout_logit).reshape(-1)
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = self.entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.sum(first_layer_kl - H1)
            return prob / num_ensemble, ARM_ori, ARM_tilde, u_list, kl_term
        else:
            return prob / num_ensemble, None, None, None, None

    def entropy(self, prob):
        return -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)

    @staticmethod
    def dropout_ori(x, logit, u):
        dropout_prob = torch.sigmoid(logit)
        z = (u < dropout_prob).float()
        return x * z

    @staticmethod
    def dropout_tilde(x, logit, u):
        dropout_prob_tilde = torch.sigmoid(-logit)
        z_tilde = (u > dropout_prob_tilde).float()
        return x * z_tilde


class BayesianVNETDetector(nn.Module):
    """
    This implements the Bayesian version of VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, length_scale: float, ensemble_num: int):
        super(BayesianVNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self.length_scale = length_scale
        self.ensemble_num = ensemble_num
        self.net = BayesianDNN(self.n_states, self.length_scale).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: Phase.TRAIN or Phase.TEST
        :returns if in Phase.TRAIN - the estimated priors [batch_size,transmission_length,n_states]
        if in Phase.TEST - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)

        if phase == Phase.TEST:
            priors, _, _, _, _ = self.net(rx, self.ensemble_num, phase)
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

            return detected_word, confidence_word
        else:
            info_for_Bayesian_training = self.net(rx, self.ensemble_num, phase)
            return info_for_Bayesian_training

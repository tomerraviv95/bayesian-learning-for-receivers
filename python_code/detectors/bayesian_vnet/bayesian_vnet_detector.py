import collections

import numpy as np
import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.utils.constants import Phase

LossVariable = collections.namedtuple('LossVariable', 'priors arm_original arm_tilde u_list kl_term')

HIDDEN1_SIZE = 200


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


def entropy(prob):
    return -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)


def dropout_ori(x, logit, u):
    dropout_prob = torch.sigmoid(logit)
    z = (u < dropout_prob).float()
    return x * z


def dropout_tilde(x, logit, u):
    dropout_prob_tilde = torch.sigmoid(-logit)
    z_tilde = (u > dropout_prob_tilde).float()
    return x * z_tilde


class BayesianDNN(nn.Module):
    """
    This Bayesian Implementation follows the paper
    "Learnable Bernoulli Dropout for Bayesian Deep Learning"
    by computing the forward pass by averaging multiple viterbinets runs, each with a different dropout mask

    To learn the dropout parameters we compute additional outputs such as KL term and output_tilde
    which will be used in the computed loss
    """

    def __init__(self, n_states: int, kl_scale: float):
        super(BayesianDNN, self).__init__()
        self.fc1 = nn.Linear(1, HIDDEN1_SIZE).to(DEVICE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, n_states).to(DEVICE)
        self.dropout_logit = nn.Parameter(torch.rand(HIDDEN1_SIZE).reshape(1, -1))
        self.T = 2 # nn.Parameter(torch.ones(1))
        self.activation = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_scale = kl_scale

    def forward(self, raw_input: torch.Tensor, num_ensemble: int, phase: Phase):
        if phase == Phase.TRAIN:
            log_probs = 0
        else:
            log_probs = 0
            probs = 0
        arm_original, arm_tilde, u_list, kl_term = [], [], [], 0

        for ind_ensemble in range(num_ensemble):
            # first layer
            x = self.activation(self.fc1(raw_input))
            u = torch.rand(x.shape).to(DEVICE)
            x_after_dropout = dropout_ori(x, self.dropout_logit, u)
            # second layer
            out = self.fc2(x_after_dropout)
            # if in train phase, keep parameters in list and compute the tilde output for arm loss calculation
            if phase == Phase.TRAIN:
                log_probs += self.log_softmax(out)
                u_list.append(u)
                # compute first variable output
                arm_original.append(self.log_softmax(out))
                # compute second variable output
                x_tilde = dropout_tilde(x, self.dropout_logit, u)
                out_tilde = self.fc2(x_tilde)
                arm_tilde.append(self.log_softmax(out_tilde))
            else:
                probs += self.softmax(out.clone().detach()/self.T)

        if phase == Phase.TRAIN:
            log_probs /= num_ensemble
        else:
            log_probs = torch.log(probs / num_ensemble)
        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)

        return LossVariable(priors=log_probs, arm_original=arm_original, arm_tilde=arm_tilde,
                            u_list=u_list, kl_term=kl_term)


class BayesianVNETDetector(nn.Module):
    """
    This implements the Bayesian version of VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, kl_scale: float, ensemble_num: int):
        super(BayesianVNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self.net = BayesianDNN(self.n_states, kl_scale).to(DEVICE)
        self.ensemble_num = ensemble_num

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

        if phase == Phase.TEST:
            priors = self.net(rx, self.ensemble_num, phase).priors
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
            return self.net(rx, self.ensemble_num, phase)

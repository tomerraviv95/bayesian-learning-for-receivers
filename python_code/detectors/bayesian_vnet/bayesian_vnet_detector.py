import numpy as np
import torch
import torch.nn as nn

from python_code import DEVICE

HIDDEN1_SIZE = 75
HIDDEN2_SIZE = 16


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
    def __init__(self, n_states, length_scale=0.1):
        super(BayesianDNN, self).__init__()
        self.fc1 = nn.Linear(1, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, n_states)
        self.dropout_logit1 = nn.parameter.Parameter(torch.tensor(2.0))
        self.dropout_logit2 = nn.parameter.Parameter(torch.tensor(2.0))
        self.activ = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.length_scale = length_scale

    def forward(self, raw_input, num_ensemble, phase):
        if phase == 'train':
            log_prob = 0
            log_prob_ARM_ori = 0
            log_prob_ARM_tilde = 0
        else:
            prob = 0
        for ind_ensemble in range(num_ensemble):
            # first layer
            u1 = torch.rand(raw_input.shape).to(DEVICE)
            x, z1 = self.dropout_ori(raw_input, self.dropout_logit1, u1)
            x = self.activ(self.fc1(x))
            if phase == 'train':
                x_tilde = self.dropout_tilde(raw_input, self.dropout_logit1, u1)
                x_tilde = self.activ(self.fc1(x_tilde))
            else:
                pass

            # second layer
            u2 = torch.rand(x.shape).to(DEVICE)
            x, z2 = self.dropout_ori(x, self.dropout_logit2, u2)
            x = self.fc2(x)
            if phase == 'train':
                x_tilde = self.dropout_tilde(x_tilde, self.dropout_logit2, u2)
                x_tilde = self.fc2(x_tilde)
            else:
                pass

            if phase == 'train':
                log_prob_ARM_ori += self.logsoftmax(x)  # .detach()
                log_prob_ARM_tilde += self.logsoftmax(x_tilde)  # .detach()
                log_prob += self.logsoftmax(x)  # training loss computed for each parameter realization
            else:
                prob += self.softmax(x)  # ensembling should be done in probability domain

        ## KL term if training
        if phase == 'train':
            # KL term
            # first layer
            first_layer_kl = self.sigmoid(self.dropout_logit1) * (self.length_scale ** 2) * (
                    torch.norm(self.fc1.weight) ** 2 + torch.norm(self.fc1.bias) ** 2) / 2
            # second layer
            second_layer_kl = self.sigmoid(self.dropout_logit2) * (self.length_scale ** 2) * (
                    torch.norm(self.fc2.weight) ** 2 + torch.norm(self.fc2.bias) ** 2) / 2

            H1 = self.entropy(self.sigmoid(self.dropout_logit1))
            H2 = self.entropy(self.sigmoid(self.dropout_logit2))
            kl_term = first_layer_kl + second_layer_kl - H1 - H2
            return log_prob / num_ensemble, log_prob_ARM_ori / num_ensemble, log_prob_ARM_tilde / num_ensemble, kl_term, (
                u1, u2)
        else:
            return torch.log(prob / num_ensemble), None, None, None, None

    def entropy(self, prob):
        return -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)

    @staticmethod
    def dropout_ori(x, logit, u):
        dropout_prob = torch.sigmoid(logit)
        z = (u < dropout_prob).float()
        return x * z, z

    @staticmethod
    def dropout_tilde(x, logit, u):
        dropout_prob_tilde = torch.sigmoid(-logit)
        z_tilde = (u > dropout_prob_tilde).float()
        return x * z_tilde


class BayesianVNETDetector(nn.Module):
    """
    This implements the Bayesian version of VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, length_scale: float, num_ensemble_tr: int,
                 num_ensemble_val: int):  # length_scale: how much we care the Bayesian prior

        super(BayesianVNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self.length_scale = length_scale
        self.num_ensemble_tr = num_ensemble_tr
        self.num_ensemble_val = num_ensemble_val
        self.net = BayesianDNN(self.n_states, self.length_scale).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        # priors = self.net(rx, num_ensemble, phase)

        if phase == 'val':
            priors, _, _, _, _ = self.net(rx, self.num_ensemble_val, phase)
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
            info_for_Bayesian_training = self.net(rx, self.num_ensemble_tr, phase)
            return info_for_Bayesian_training

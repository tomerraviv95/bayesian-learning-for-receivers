import collections

import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_USER, N_ANT
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()

HIDDEN_BASE_SIZE = 256

LossVariable = collections.namedtuple('LossVariable', 'priors arm_original arm_tilde u_list kl_term dropout_logit')


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


class BayesianDeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(in_features=s_nK+s_nN-1, out_features=60, bias=True)
      (sigmoid): Sigmoid()
      (fullyConnectedLayer): Linear(in_features=60, out_features=30, bias=True)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(in_features=30, out_features=2, bias=True)
    ================================
    Note:
    The output of the network is not probabilities,
    to obtain probabilities apply a softmax function to the output, viz.
    output = DeepSICNet(data)
    probs = torch.softmax(output, dim), for a batch inference, set dim=1; otherwise dim=0.
    """

    def __init__(self, ensemble_num, kl_scale):
        super(BayesianDeepSICDetector, self).__init__()
        classes_num = BPSKModulator.constellation_size
        hidden_size = HIDDEN_BASE_SIZE * classes_num
        linear_input = (classes_num // 2) * N_ANT + (classes_num - 1) * (N_USER - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, classes_num)
        self.num_ensemble = ensemble_num
        self.kl_scale = kl_scale
        self.dropout_logit = nn.Parameter(torch.rand(hidden_size).reshape(1, -1))
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.T = 1

    def forward(self, rx: torch.Tensor, phase: Phase) -> torch.Tensor:
        log_probs = 0
        arm_original, arm_tilde, u_list, kl_term = [], [], [], 0

        for ind_ensemble in range(self.num_ensemble):
            # first layer
            x = self.activation(self.fc1(rx))
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
                log_probs += self.log_softmax(out / self.T)

        log_probs /= self.num_ensemble

        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)

        return LossVariable(priors=log_probs, arm_original=arm_original, arm_tilde=arm_tilde,
                            u_list=u_list, kl_term=kl_term, dropout_logit=self.dropout_logit)

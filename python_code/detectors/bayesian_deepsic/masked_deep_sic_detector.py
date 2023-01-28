import collections

import torch
from torch import nn

from python_code import DEVICE
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()

LossVariable = collections.namedtuple('LossVariable', 'priors u arm_original arm_tilde dropout_logit out_tilde kl_term')


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


class MaskedDeepSICDetector(nn.Module):
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

    def __init__(self, linear_input, hidden_size, classes_num, kl_scale):
        super(MaskedDeepSICDetector, self).__init__()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)
        self.kl_scale = kl_scale
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, rx: torch.Tensor, dropout_logit: torch.Tensor, phase: Phase = Phase.TEST) -> LossVariable:
        kl_term = 0
        # first layer
        x = self.activation(self.fc1(rx))
        u = torch.rand(x.shape).to(DEVICE)
        x_after_dropout = dropout_ori(x, dropout_logit, u)
        # second layer
        out = self.fc2(x_after_dropout)
        # tilde
        x_tilde = dropout_tilde(x, dropout_logit, u)
        out_tilde = self.fc2(x_tilde)

        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=self.log_softmax(out), u=u,
                                arm_original=self.log_softmax(out),
                                arm_tilde=self.log_softmax(out_tilde),
                                dropout_logit=dropout_logit,
                                out_tilde=out_tilde, kl_term=kl_term)

        return self.log_softmax(out)

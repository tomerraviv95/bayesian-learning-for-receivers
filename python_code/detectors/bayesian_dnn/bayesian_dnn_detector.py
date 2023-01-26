import collections

import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
from python_code.utils.trellis_utils import calculate_symbols_from_states

conf = Config()

LossVariable = collections.namedtuple('LossVariable', 'priors u arm_original arm_tilde dropout_logit out_tilde kl_term')


class BayesianDNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant, n_states, hidden_size):
        super(BayesianDNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(MODULATION_NUM_MAPPING[conf.modulation_type] * self.n_user // 2, self.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.hidden_size, self.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.hidden_size, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        out = self.net(rx)

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

        if phase == Phase.TEST:
            # Decode the output
            estimated_states = torch.argmax(out, dim=1)
            estimated_words = calculate_symbols_from_states(self.n_ant, estimated_states)
            return estimated_words.long()
        else:
            return out

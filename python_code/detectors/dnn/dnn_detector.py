import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config

conf = Config()

HIDDEN_SIZE = 60


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant):
        super(DNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** n_ant
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(MODULATION_NUM_MAPPING[conf.modulation_type] * self.n_user // 2, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        return self.net(rx)

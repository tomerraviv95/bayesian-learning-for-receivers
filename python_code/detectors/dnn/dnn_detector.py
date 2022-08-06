import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_symbols_from_states

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
        self.n_states = BPSKModulator.constellation_size ** n_ant
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(self.n_user, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        out = self.net(rx)
        if phase == 'val':
            # Decode the output
            estimated_states = torch.argmax(out, dim=1)
            estimated_words = calculate_symbols_from_states(self.n_ant, estimated_states)
            return estimated_words.long()
        else:
            return out

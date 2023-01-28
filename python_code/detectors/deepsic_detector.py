import torch
from torch import nn

from python_code.channel.channels_hyperparams import N_USER, N_ANT, MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType

conf = Config()

HIDDEN_BASE_SIZE = 64


class DeepSICDetector(nn.Module):
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

    def __init__(self):
        super(DeepSICDetector, self).__init__()
        classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        hidden_size = HIDDEN_BASE_SIZE * classes_num
        base_rx_size = N_ANT if conf.modulation_type == ModulationType.BPSK.name else 2 * N_ANT
        linear_input = base_rx_size + (classes_num - 1) * (N_USER - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        out0 = self.activation(self.fc1(rx))
        out1 = self.fc2(out0)
        return out1

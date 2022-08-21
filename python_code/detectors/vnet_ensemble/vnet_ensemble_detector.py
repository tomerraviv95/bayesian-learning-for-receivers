import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.detectors.vnet.vnet_detector import VNETDetector
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()


class VNETEnsembleDetector(nn.Module):
    """
    This implements the VA Ensemble decoder
    """

    def __init__(self, n_states: int, alpha=1):

        super(VNETEnsembleDetector, self).__init__()
        self.alpha = alpha
        self.n_states = n_states
        self.detector = VNETDetector(n_states=n_states, dropout_rate=conf.dropout_rate)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        out = torch.zeros([rx.shape[0], self.n_states]).to(DEVICE)
        for i in range(self.alpha):
            cur_out = self.detector(rx, phase=phase)
            out += cur_out
        if phase == Phase.TEST:
            return ((out / self.alpha) > 0.5).type(out.dtype)
        else:
            return out / self.alpha

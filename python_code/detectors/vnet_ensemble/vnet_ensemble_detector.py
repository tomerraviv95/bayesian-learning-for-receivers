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
        self._alpha = alpha
        self._n_states = n_states
        self._detector = VNETDetector(n_states=self._n_states, dropout_rate=conf.dropout_rate)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the Ensemble ViterbiNet algorithm - MC dropout
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        if phase == Phase.TEST:
            out = torch.zeros([rx.shape[0], rx.shape[1]]).to(DEVICE)
            for i in range(self._alpha):
                cur_out = self._detector(rx, phase=phase)
                out += cur_out
            out /= self._alpha
            return (out > 0.5).type(out.dtype)
        else:
            return self._detector(rx, phase=phase)
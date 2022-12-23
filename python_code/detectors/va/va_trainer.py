import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.trainer import Trainer
from python_code.detectors.va.va_detector import VADetector
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()


class VATrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = BPSKModulator.constellation_size ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.probs_vec = None
        super().__init__()

    def __str__(self):
        return 'Viterbi Algorithm'

    def _initialize_detector(self):
        """
        Loads the Viterbi detector
        """
        self.detector = VADetector(n_states=self.n_states, memory_length=self.memory_length)

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase=Phase.TEST, h=h)
        return detected_word

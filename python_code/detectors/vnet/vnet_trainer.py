import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.trainer import Trainer
from python_code.detectors.vnet.vnet_detector import VNETDetector
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
from python_code.utils.trellis_utils import calculate_siso_states

conf = Config()
EPOCHS = 500


class VNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet seq_model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = BPSKModulator.constellation_size ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.lr = 5e-3
        self.probs_vec = None
        super().__init__()

    def __str__(self):
        return 'ViterbiNet'

    def _initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states)

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1,transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(self.memory_length, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase=Phase.TEST)
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.deep_learning_setup(self.lr)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss

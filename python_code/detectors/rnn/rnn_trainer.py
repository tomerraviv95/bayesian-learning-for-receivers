from random import randint

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.detectors.rnn.rnn_detector import RNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_siso_states

conf = Config()
EPOCHS = 500
BATCH_SIZE = 16


class RNNTrainer(Trainer):
    """
    Trainer for the RNNTrainer model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = 2 ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.probs_vec = None
        self.lr = 1e-2
        super().__init__()

    def __str__(self):
        return 'RNN Detector'

    def _initialize_detector(self):
        """
        Loads the RNN detector
        """
        self.detector = RNNDetector()

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

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase='val')
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if conf.from_scratch:
            self._initialize_detector()
        self.deep_learning_setup()

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            word_ind = randint(a=0, b=conf.online_repeats_n)
            subword_ind = randint(a=0, b=conf.pilot_size - BATCH_SIZE)
            ind = word_ind * conf.pilot_size + subword_ind
            # pass through detector
            soft_estimation = self.detector(rx[ind: ind + BATCH_SIZE].float(), phase='train')
            current_loss = self.run_train_loop(est=soft_estimation,
                                               tx=tx[ind:ind + BATCH_SIZE])
            loss += current_loss

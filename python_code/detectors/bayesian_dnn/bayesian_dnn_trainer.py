from typing import List

import torch

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER, MODULATION_NUM_MAPPING
from python_code.detectors.bayesian_dnn.bayesian_dnn_detector import LossVariable, BayesianDNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, ModulationType, HALF
from python_code.utils.trellis_utils import calculate_mimo_states, get_bits_from_qpsk_symbols, \
    get_qpsk_symbols_from_bits, calculate_symbols_from_states

conf = Config()

EPOCHS = 400
HIDDEN_SIZE = 60


class BayesianDNNTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.lr = 5e-3
        self.ensemble_num = 5
        self.kl_scale = 5
        self.kl_beta = 1e-2
        self.arm_beta = 1
        self.n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** self.n_ant
        super().__init__()

    def __str__(self):
        return 'Bayesian DNN Detector'

    def _initialize_detector(self):
        """
            Loads the DNN detector
        """
        self.detector = BayesianDNNDetector(n_user=self.n_user, n_ant=self.n_ant, n_states=self.n_states,
                                            hidden_size=HIDDEN_SIZE, kl_scale=self.kl_scale,
                                            ensemble_num=self.ensemble_num)

    def calc_loss(self, est: List[LossVariable], tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        gt_states = calculate_mimo_states(self.n_ant, tx).to(DEVICE)
        data_fitting_loss_term = self.criterion(input=est.priors, target=gt_states)
        loss = data_fitting_loss_term
        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=est.arm_original[i], target=gt_states)
            loss_term_arm_tilde = self.criterion(input=est.arm_tilde[i], target=gt_states)
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (est.u_list[i] - HALF)
            arm_loss += torch.matmul(grad_logit, est.dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += self.arm_beta * arm_loss + kl_term
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        if conf.modulation_type == ModulationType.BPSK.name:
            rx = rx.float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        soft_estimation = self.detector(rx, phase=Phase.TEST).priors
        estimated_states = torch.argmax(soft_estimation, dim=1)
        detected_word = calculate_symbols_from_states(self.n_ant, estimated_states).long()

        if conf.modulation_type == ModulationType.QPSK.name:
            detected_word = get_bits_from_qpsk_symbols(detected_word)
            confidence_bits = detected_word
            confidence_word = get_qpsk_symbols_from_bits(detected_word)

        return detected_word, (confidence_bits, confidence_word)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.deep_learning_setup(self.lr)

        if conf.modulation_type == ModulationType.QPSK.name:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss

from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.model_based_bayesian_deepsic.bayesian_deep_sic_detector import LossVariable, BayesianDeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, Phase

conf = Config()
ITERATIONS = 2
EPOCHS = 400


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


class ModelBasedBayesianDeepSICTrainer(Trainer):
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
        super().__init__()

    def __str__(self):
        return 'Model-Based Bayesian DeepSIC'

    def _initialize_detector(self):
        self.detector = [
            [BayesianDeepSICDetector(self.ensemble_num, self.kl_scale).to(DEVICE) for _ in range(ITERATIONS)] for _ in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=est.priors, target=tx.long())
        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=est.arm_original[i], target=tx.long())
            loss_term_arm_tilde = self.criterion(input=est.arm_tilde[i], target=tx.long())
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (est.u_list[i] - HALF)
            arm_loss += torch.matmul(grad_logit, est.dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += self.arm_beta * arm_loss + kl_term
        return loss

    @staticmethod
    def preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        loss = 0
        y_total = self.preprocess(rx)
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total, phase=Phase.TRAIN)
            current_loss = self.run_train_loop(soft_estimation, tx)
            loss += current_loss

    def train_models(self, model: List[List[BayesianDeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        initial_probs = tx.clone()
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx, Phase.TRAIN)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)

    def forward(self, rx: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        probs_vec = HALF * torch.ones(conf.block_length - conf.pilot_size, N_ANT).to(DEVICE).float()
        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx, phase=Phase.TEST)
        detected_word = BPSKModulator.demodulate(prob_to_BPSK_symbol(probs_vec.float()))
        new_probs_vec = torch.cat([probs_vec.unsqueeze(dim=2), (1 - probs_vec).unsqueeze(dim=2)], dim=2)
        confident_bits = 1 - torch.argmax(new_probs_vec, dim=2)
        confidence_word = torch.amax(new_probs_vec, dim=2)
        return detected_word, (confident_bits, confidence_word)

    def prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor, phase: Phase) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](preprocessed_input, phase).priors)
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

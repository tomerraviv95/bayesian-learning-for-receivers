from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER, MODULATION_NUM_MAPPING
from python_code.channel.modulator import BPSKModulator, QPSKModulator
from python_code.detectors.model_based_bayesian_deepsic.bayesian_deep_sic_detector import LossVariable, BayesianDeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, Phase, ModulationType, QUARTER
from python_code.utils.trellis_utils import prob_to_QPSK_symbol

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
        if conf.modulation_type == ModulationType.BPSK.name:
            return rx.float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            y_input = torch.view_as_real(rx[:, :N_ANT]).float().reshape(rx.shape[0], -1)
            return torch.cat([y_input, rx[:, N_ANT:].float()], dim=1)

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

    def _initialize_probs(self, tx):
        if conf.modulation_type == ModulationType.BPSK.name:
            initial_probs = tx.clone()
        elif conf.modulation_type == ModulationType.QPSK.name:
            initial_probs = torch.zeros(tx.shape).to(DEVICE).unsqueeze(-1).repeat(
                [1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1])
            relevant_inds = []
            for i in range(MODULATION_NUM_MAPPING[conf.modulation_type] - 1):
                relevant_ind = (tx == i + 1)
                relevant_inds.append(relevant_ind.unsqueeze(-1))
            relevant_inds = torch.cat(relevant_inds, dim=2)
            initial_probs[relevant_inds] = 1
        else:
            raise ValueError("No such constellation!")
        return initial_probs

    def _initialize_probs_for_training(self, tx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        elif conf.modulation_type == ModulationType.QPSK.name:
            probs_vec = QUARTER * torch.ones(tx.shape).to(DEVICE).unsqueeze(-1).repeat(
                [1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1])
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        initial_probs = self._initialize_probs(tx)
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx, Phase.TRAIN)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)

    def _initialize_probs_for_infer(self):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(conf.block_length - conf.pilot_size, N_ANT).to(DEVICE).float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            probs_vec = QUARTER * torch.ones((conf.block_length - 2 * conf.pilot_size) // 2, N_ANT).to(
                DEVICE).unsqueeze(-1).repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1]).float()
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def forward(self, rx: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        probs_vec = self._initialize_probs_for_infer()
        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx, phase=Phase.TEST)
        confidence_word, confident_bits, detected_word = self.compute_output(probs_vec)
        return detected_word, (confident_bits, confidence_word)

    def compute_output(self, probs_vec):
        if conf.modulation_type == ModulationType.BPSK.name:
            detected_word = BPSKModulator.demodulate(prob_to_BPSK_symbol(probs_vec.float()))
            new_probs_vec = torch.cat([probs_vec.unsqueeze(dim=2), (1 - probs_vec).unsqueeze(dim=2)], dim=2)
            confident_bits = 1 - torch.argmax(new_probs_vec, dim=2)
            confidence_word = torch.amax(new_probs_vec, dim=2)
        elif conf.modulation_type == ModulationType.QPSK.name:
            detected_word = QPSKModulator.demodulate(prob_to_QPSK_symbol(probs_vec.float()))
            new_probs_vec = torch.cat([probs_vec, (1 - probs_vec.sum(dim=2)).unsqueeze(dim=2)], dim=2)
            confident_bits = detected_word
            confidence_word = torch.amax(new_probs_vec, dim=2)
        else:
            raise ValueError("No such constellation!")
        return confidence_word, confident_bits, detected_word

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

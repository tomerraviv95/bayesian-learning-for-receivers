from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER, MODULATION_NUM_MAPPING
from python_code.channel.modulator import BPSKModulator, QPSKModulator
from python_code.detectors.bayesian_deepsic.masked_deep_sic_detector import LossVariable, \
    MaskedDeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, Phase, ModulationType
from python_code.utils.trellis_utils import prob_to_QPSK_symbol

conf = Config()
ITERATIONS = 2
EPOCHS = 400

BASE_HIDDEN_SIZE = 64


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


class BayesianDeepSICTrainer(Trainer):
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
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        self.hidden_size = BASE_HIDDEN_SIZE * self.classes_num
        self.linear_input = (self.classes_num // 2) * N_ANT + (self.classes_num - 1) * (
                N_USER - 1)  # from DeepSIC paper
        self.T = 1
        super().__init__()

    def __str__(self):
        return 'Bayesian DeepSIC'

    def _initialize_detector(self):
        detectors_list = [
            [MaskedDeepSICDetector(self.linear_input, self.hidden_size, self.classes_num, self.kl_scale).to(DEVICE) for
             _ in range(ITERATIONS)]
            for _ in
            range(self.n_user)]  # 2D list for Storing the DeepSIC Networks
        flat_detectors_list = [detector for sublist in detectors_list for detector in sublist]
        self.detector = nn.ModuleList(flat_detectors_list)
        dropout_logits_list = [
            [nn.Parameter(torch.rand(self.hidden_size).reshape(1, -1)).to(DEVICE) for _ in range(ITERATIONS)] for _ in
            range(self.n_user)]  # 2D list for Storing the dropout logits
        self.dropout_logits = [dropout_logit for sublist in dropout_logits_list for dropout_logit in sublist]

    def calc_loss(self, est: List[List[LossVariable]], tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = 0
        for user in range(self.n_user):
            for ind_ensemble in range(self.ensemble_num):
                cur_loss_var = est[user][ind_ensemble]
                cur_tx = tx[user]
                # point loss
                loss += self.criterion(input=cur_loss_var.priors, target=cur_tx.long()) / self.ensemble_num
                # ARM Loss
                loss_term_arm_original = self.criterion(input=cur_loss_var.arm_original, target=cur_tx.long())
                loss_term_arm_tilde = self.criterion(input=cur_loss_var.arm_tilde, target=cur_tx.long())
                arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
                grad_logit = arm_delta * (cur_loss_var.u - HALF)
                arm_loss = torch.matmul(grad_logit, cur_loss_var.dropout_logit.T)
                arm_loss = torch.mean(arm_loss)
                # KL Loss
                kl_term = self.kl_beta * cur_loss_var.kl_term
                loss += self.arm_beta * arm_loss + kl_term
        return loss

    @staticmethod
    def preprocess(rx: torch.Tensor) -> torch.Tensor:
        if conf.modulation_type == ModulationType.BPSK.name:
            return rx.float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            y_input = torch.view_as_real(rx[:, :N_ANT]).float().reshape(rx.shape[0], -1)
            return torch.cat([y_input, rx[:, N_ANT:].float()], dim=1)

    def infer_model(self, single_model: nn.Module, dropout_logit: nn.Parameter, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        return single_model(y_total, dropout_logit, Phase.TRAIN)

    def infer_models(self, rx_all: List[torch.Tensor]):
        loss_vars = []
        for user in range(self.n_user):
            loss_var = self.infer_model(self.detector[user * ITERATIONS + ITERATIONS - 1],
                                        self.dropout_logits[user * ITERATIONS + ITERATIONS - 1],
                                        rx_all[user])
            loss_vars.append(loss_var)
        return loss_vars

    def _initialize_probs_for_training(self, tx):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        elif conf.modulation_type == ModulationType.QPSK.name:
            probs_vec = 1 / 4 * torch.ones(tx.shape).to(DEVICE).unsqueeze(-1).repeat(
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
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            total_loss_vars = [[] for user in range(self.n_user)]
            for ind_ensemble in range(self.ensemble_num):
                # Initializing the probabilities
                probs_vec = self._initialize_probs_for_training(tx)
                # Training the DeepSICNet for each user-symbol/iteration
                for i in range(ITERATIONS):
                    # Generating soft symbols for training purposes
                    probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx)
                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
                loss_vars = self.infer_models(rx_all)
                for user in range(self.n_user):
                    total_loss_vars[user].append(loss_vars[user])
            loss = self.calc_loss(total_loss_vars, tx_all)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _initialize_probs_for_infer(self):
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(conf.block_length - conf.pilot_size, N_ANT).to(DEVICE).float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            probs_vec = 1 / 4 * torch.ones((conf.block_length - 2 * conf.pilot_size) // 2, N_ANT).to(
                DEVICE).unsqueeze(-1).repeat([1, 1, MODULATION_NUM_MAPPING[conf.modulation_type] - 1]).float()
        else:
            raise ValueError("No such constellation!")
        return probs_vec

    def forward(self, rx: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        total_probs_vec = 0
        for ind_ensemble in range(self.ensemble_num):
            # detect and decode
            probs_vec = self._initialize_probs_for_infer()
            for i in range(ITERATIONS):
                probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            total_probs_vec += probs_vec
        total_probs_vec /= self.ensemble_num
        confidence_word, confident_bits, detected_word = self.compute_output(total_probs_vec)
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

    def calculate_posteriors(self, model: nn.ModuleList, i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            output = self.softmax(
                model[user * ITERATIONS + i - 1](preprocessed_input, self.dropout_logits[user * ITERATIONS + i - 1],
                                                 Phase.TEST).priors)
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

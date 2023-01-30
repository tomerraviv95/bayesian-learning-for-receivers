## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.detectors.deepsic.bayesian_deepsic.masked_deep_sic_detector import LossVariable, \
    MaskedDeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import HALF, Phase, ModulationType

ITERATIONS = 2
EPOCHS = 400

BASE_HIDDEN_SIZE = 64


class BayesianDeepSICTrainer(DeepSICTrainer):
    """
    The Black-Box Bayesian Approach Applied to DeepSIC
    """
    def __init__(self):
        self.ensemble_num = 5
        self.kl_scale = 5
        self.kl_beta = 1e-2
        self.arm_beta = 1
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        self.hidden_size = BASE_HIDDEN_SIZE * self.classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        self.linear_input = base_rx_size + (self.classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
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
                grad_logit = arm_delta * (cur_loss_var.u_list - HALF)
                arm_loss = torch.matmul(grad_logit, cur_loss_var.dropout_logit.T)
                arm_loss = torch.mean(arm_loss)
                # KL Loss
                kl_term = self.kl_beta * cur_loss_var.kl_term
                loss += self.arm_beta * arm_loss + kl_term
        return loss

    def infer_model(self, single_model: nn.Module, dropout_logit: nn.Parameter, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        return single_model(y_total, dropout_logit, phase=Phase.TRAIN)

    def infer_models(self, rx_all: List[torch.Tensor]):
        loss_vars = []
        for user in range(self.n_user):
            loss_var = self.infer_model(self.detector[user * ITERATIONS + ITERATIONS - 1],
                                        self.dropout_logits[user * ITERATIONS + ITERATIONS - 1],
                                        rx_all[user])
            loss_vars.append(loss_var)
        return loss_vars

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
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
                model[user * ITERATIONS + i - 1](preprocessed_input, self.dropout_logits[user * ITERATIONS + i - 1]))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

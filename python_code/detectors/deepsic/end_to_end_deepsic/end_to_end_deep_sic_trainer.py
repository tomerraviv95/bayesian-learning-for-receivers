from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer

ITERATIONS = 2
EPOCHS = 400


class EndToEndDeepSICTrainer(DeepSICTrainer):
    def __init__(self):

        super().__init__()

    def __str__(self):
        return 'End-To-End DeepSIC'

    def _initialize_detector(self):
        detectors_list = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                          range(self.n_user)]  # 2D list for Storing the DeepSIC Networks
        flat_detectors_list = [detector for sublist in detectors_list for detector in sublist]
        self.detector = nn.ModuleList(flat_detectors_list)

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        y_total = self.preprocess(rx)
        soft_estimation = single_model(y_total)
        loss = self.calc_loss(est=soft_estimation, tx=tx.int())
        return loss

    def train_models(self, tx_all: List[torch.Tensor], rx_all: List[torch.Tensor]):
        cur_loss = 0
        for user in range(self.n_user):
            cur_loss += self.train_model(self.detector[user * ITERATIONS + ITERATIONS - 1], tx_all[user], rx_all[user])
        return cur_loss

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training the entire networks by end-to-end manner.
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            # Initializing the probabilities
            probs_vec = self._initialize_probs_for_training(tx)
            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(ITERATIONS):
                # Generating soft symbols for training purposes
                probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            loss = self.train_models(tx_all, rx_all)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
            output = self.softmax(model[user * ITERATIONS + i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer

ITERATIONS = 2
EPOCHS = 400


class SeqDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        self.detector = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

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
            soft_estimation = single_model(y_total)
            current_loss = self.run_train_loop(soft_estimation, tx)
            loss += current_loss

    def train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
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

        initial_probs = self._initialize_probs(tx)
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)

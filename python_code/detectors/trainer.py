import random
from typing import Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import RMSprop, Adam, SGD

from python_code import DEVICE
from python_code.augmentations.augmenter_wrapper import AugmenterWrapper
from python_code.augmentations.augmentations_plotting_utils import online_plotting
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.config_singleton import Config
from python_code.utils.metrics import calculate_ber

conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """
    def __init__(self):
        # initialize matrices, datasets and detector
        self._initialize_dataloader()
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.detector = None

    # calculate train loss
    def calc_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=self.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=self.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.val_block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num)
        self.dataloader = torch.utils.data.DataLoader(self.channel_dataset)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def init_priors(self):
        """
        DeepSIC employs this initialization
        """
        pass

    def evaluate(self) -> Union[float, np.ndarray]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: np.ndarray
        """
        print(conf.aug_type)
        total_ser = 0
        # draw words for a given snr
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
        # either None or in case of DeepSIC intializes the priors
        self.init_priors()
        ser_by_word = np.zeros(transmitted_words.shape[0])
        # initialize the augmentations class instance
        augmenter_wrapper = AugmenterWrapper(conf.aug_type, conf.fading_in_channel)
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            # get current word and channel
            tx, h, rx = transmitted_words[block_ind], hs[block_ind], received_words[block_ind]
            # split words into data and pilot part
            tx_pilot, tx_data = tx[:conf.pilot_size], tx[conf.pilot_size:]
            rx_pilot, rx_data = rx[:conf.pilot_size], rx[conf.pilot_size:]
            if conf.is_online_training:
                # augment received words by the number of desired repeats
                augmenter_wrapper.update_hyperparams(rx_pilot, tx_pilot)
                y_aug, x_aug = augmenter_wrapper.augment_batch(h, rx_pilot, tx_pilot)
                # re-train the detector
                self._online_training(x_aug, y_aug)
            # detect data part after training on the pilot part
            detected_word = self.forward(rx_data, self.probs_vec)
            # calculate accuracy
            ser = calculate_ber(detected_word, tx_data[:, :rx.shape[1]])
            print('*' * 20)
            print(f'current: {block_ind, ser}')
            total_ser += ser
            ser_by_word[block_ind] = ser
            self.init_priors()

        total_ser /= conf.blocks_num
        print(f'Final ser: {total_ser}')
        return total_ser

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def plot_regions(self):
        """
        Used in the augmentations plotting method under plotters module. Used for drawing the relevant figures for
        the paper.
        """
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
        augmenter_wrapper = AugmenterWrapper(conf.aug_type, conf.fading_in_channel)
        for block_ind in range(conf.blocks_num):
            # get current word and channel
            transmitted_word = transmitted_words[block_ind]
            h = hs[block_ind]
            received_word = received_words[block_ind]
            # split words into data and pilot part
            x_pilot, x_data = transmitted_word[:conf.pilot_size], transmitted_word[conf.pilot_size:]
            y_pilot, y_data = received_word[:conf.pilot_size], received_word[conf.pilot_size:]
            # augment received words by the number of desired repeats
            augmenter_wrapper.update_hyperparams(y_pilot, x_pilot)
            y_aug, x_aug = augmenter_wrapper.augment_batch(h, y_pilot, x_pilot)
            # if online training flag is on - train using pilots part
            online_plotting(x_aug, y_aug, h)

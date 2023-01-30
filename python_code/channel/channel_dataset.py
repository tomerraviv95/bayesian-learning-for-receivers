import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE, conf
from python_code.channel.mimo_channels.mimo_channel_dataset import MIMOChannel
from python_code.utils.constants import ModulationType
from python_code.utils.python_utils import normalize_for_modulation



class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self, block_length: int, pilots_length: int, blocks_num: int):
        self.blocks_num = blocks_num
        self.block_length = block_length
        self.channel_type = MIMOChannel(block_length, pilots_length)

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        tx_full = np.empty((self.blocks_num, normalize_for_modulation(self.block_length), self.channel_type.tx_length))
        h_full = np.empty((self.blocks_num, *self.channel_type.h_shape))
        rx_full = np.empty((self.blocks_num, normalize_for_modulation(self.block_length), self.channel_type.rx_length),
                           dtype=complex
                           if conf.modulation_type in [ModulationType.QPSK.name, ModulationType.EightPSK.name]
                           else float)
        # accumulate words until reaches desired number
        for index in range(self.blocks_num):
            tx, h, rx = self.channel_type.get_vectors(snr, index)
            # accumulate
            tx_full[index] = tx
            rx_full[index] = rx
            h_full[index] = h

        database.append((tx_full, rx_full, h_full))

    def __getitem__(self, snr_list: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, database) for snr in snr_list]
        tx, rx, h = (np.concatenate(arrays) for arrays in zip(*database))
        tx, rx, h = torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(device=DEVICE), torch.Tensor(
            h).to(device=DEVICE)
        return tx, rx, h

    def __len__(self):
        return self.block_length

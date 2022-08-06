import torch

from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType
from python_code.utils.trellis_utils import get_bits_from_qpsk_symbols

conf = Config()


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    if conf.modulation_type == ModulationType.QPSK.name:
        target = get_bits_from_qpsk_symbols(target)
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    return 1 - bits_acc

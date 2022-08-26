import torch

from python_code.utils.config_singleton import Config

conf = Config()


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    equal_vec = torch.eq(prediction, target).float()
    bits_acc = torch.mean(equal_vec).item()
    return 1 - bits_acc

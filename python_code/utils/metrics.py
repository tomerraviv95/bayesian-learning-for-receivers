import numpy as np
import torch

from python_code.utils.config_singleton import Config

conf = Config()


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    return 1 - bits_acc


def calculate_reliability_and_ece(correct_values_list, error_values_list, values):
    correct_values_list, error_values_list = np.array(correct_values_list), np.array(error_values_list)
    avg_confidence_per_bin, avg_acc_per_bin, inbetween_indices_number_list = [], [], []
    for val_j, val_j_plus_1 in zip(values[:-1], values[1:]):
        avg_confidence_value_in_bin, avg_acc_value_in_bin = 0, 0
        inbetween_correct_indices = np.logical_and(val_j <= correct_values_list,
                                                   correct_values_list <= val_j_plus_1)
        inbetween_errored_indices = np.logical_and(val_j <= error_values_list, error_values_list <= val_j_plus_1)
        inbetween_indices_number = inbetween_correct_indices.sum() + inbetween_errored_indices.sum()
        if inbetween_indices_number > 0:
            correct_values = correct_values_list[inbetween_correct_indices]
            errored_values = error_values_list[inbetween_errored_indices]
            avg_acc_value_in_bin = len(correct_values) / (len(correct_values) + len(errored_values))
            avg_confidence_value_in_bin = np.mean(np.concatenate([correct_values, errored_values]))
        avg_acc_per_bin.append(avg_acc_value_in_bin)
        avg_confidence_per_bin.append(avg_confidence_value_in_bin)
        inbetween_indices_number_list.append(inbetween_indices_number)
        print(
            f'Bin range:({val_j, val_j_plus_1}), Total:{inbetween_indices_number}, Avg confidence:{avg_confidence_value_in_bin}, Avg accuracy:{avg_acc_value_in_bin}')
    # calculate ECE
    confidence_acc_diff = np.abs(np.array(avg_confidence_per_bin) - np.array(avg_acc_per_bin))
    ece_measure = np.sum(np.array(inbetween_indices_number_list) * confidence_acc_diff) / sum(
        inbetween_indices_number_list)
    return avg_acc_per_bin, avg_confidence_per_bin, ece_measure

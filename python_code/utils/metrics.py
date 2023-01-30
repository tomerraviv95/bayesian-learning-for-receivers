from itertools import chain
from typing import List, Tuple

import numpy as np
import torch

SENSITIVITY = 1e-3


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    return 1 - bits_acc


def calculate_reliability_and_ece(correct_values_list: List[float], error_values_list: List[float],
                                  values: List[float]) -> Tuple[List[float], List[float], float, List[int]]:
    """
    Input is two lists, of the correctly detected and incorrectly detected confidence values
    Computes the two lists of accuracy and confidences (red and blue bar plots in paper), the ECE measure and the
    normalized frequency count per bin (green bar plot in paper)
    """
    correct_values_list = np.array(list(chain.from_iterable(correct_values_list)))
    error_values_list = np.array(list(chain.from_iterable(error_values_list)))
    avg_confidence_per_bin, avg_acc_per_bin, inbetween_indices_number_list = [], [], []
    total_values = len(correct_values_list) + len(error_values_list)
    print(total_values)
    # calculate the mean accuracy and mean confidence for the given range
    for val_j, val_j_plus_1 in zip(values[:-1], values[1:]):
        avg_confidence_value_in_bin, avg_acc_value_in_bin = 0, 0
        inbetween_correct_indices = np.logical_and(val_j <= correct_values_list,
                                                   correct_values_list <= val_j_plus_1)
        inbetween_errored_indices = np.logical_and(val_j <= error_values_list, error_values_list <= val_j_plus_1)
        inbetween_indices_number = inbetween_correct_indices.sum() + inbetween_errored_indices.sum()
        if total_values * SENSITIVITY < inbetween_indices_number:
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
    # calculate the normalized samples frequency
    samples_per_bin = np.array(inbetween_indices_number_list)
    normalized_samples_per_bin = samples_per_bin / np.sum(samples_per_bin)
    return avg_acc_per_bin, avg_confidence_per_bin, ece_measure, normalized_samples_per_bin

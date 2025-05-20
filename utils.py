import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.utils import resample


def uniform_subsample_df(df, column, n_bins=50, samples_per_bin=None):
    """
    Subsample a DataFrame to create a more uniform distribution of a specified column.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Name of the column to make uniform
    n_bins : int
        Number of bins to use for discretizing the column
    samples_per_bin : int or None
        Number of samples to keep per bin. If None, uses the minimum count across bins

    Returns:
    --------
    pandas.DataFrame
        Subsampled DataFrame with more uniform distribution of specified column
    """
    # Get the column values
    y = df[column].values

    # Create bins
    bins = np.linspace(y.min(), y.max(), n_bins + 1)

    # Find which bin each sample belongs to
    bin_indices = np.digitize(y, bins) - 1

    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    # If samples_per_bin not specified, use minimum non-zero count
    if samples_per_bin is None:
        samples_per_bin = max(1, min(count for count in bin_counts if count > 0))
    # Initialize list for subsampled data
    subsampled_dfs = []

    # Subsample from each bin
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            # Get samples from this bin
            df_bin = df[mask]

            # If we have more samples than needed, subsample
            if len(df_bin) > samples_per_bin:
                df_bin = df_bin.sample(n=samples_per_bin, random_state=42)
            subsampled_dfs.append(df_bin)

    # Concatenate all subsampled data
    return pd.concat(subsampled_dfs, axis=0).reset_index(drop=True)


def balanced_subsample(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a balanced subsample of binary classification data by randomly sampling from the larger class
    to match the size of the smaller class.

    :param y_true: Array of true binary labels (0 or 1)
    :param y_prob: Array of predicted probabilities corresponding to the true labels
    :return: Tuple of (balanced_y_true, balanced_y_prob) containing equal numbers of positive and negative samples
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # Determine size of smaller class
    n_samples = min(len(pos_idx), len(neg_idx))

    # Subsample the larger class to match the smaller class size
    if len(pos_idx) > n_samples:
        pos_idx = resample(pos_idx, n_samples=n_samples, random_state=42)
    if len(neg_idx) > n_samples:
        neg_idx = resample(neg_idx, n_samples=n_samples, random_state=42)

    # Combine indices and sort them to maintain original order
    balanced_idx = np.sort(np.concatenate([pos_idx, neg_idx]))

    # Update arrays with balanced samples
    y_true = y_true[balanced_idx]
    y_prob = y_prob[balanced_idx]
    return y_true, y_prob


def bin_calibrate(
    y_pred: np.ndarray, y_cal: np.ndarray, B: int
) -> Tuple[np.ndarray, np.ndarray]:
    """fixed width bin calibration with B bins
    :param y_pred: predictions
    :param y_cal: target
    :param B: number of bins
    :return: calibrated intervals and values
    """

    sorted_y = np.asarray(y_cal)[np.argsort(y_pred)]
    scores = np.asarray(y_pred)[np.argsort(y_pred)]
    delta = int(len(y_cal) + 1) / B
    intervals = []
    new_values = []
    for i in range(B):
        if i == 0:
            intervals.append((0, scores[int((i + 1) * delta)]))
        elif i == B - 1:
            intervals.append((scores[int(i * delta)], 1))
        else:
            intervals.append((scores[int(i * delta)], scores[int((i + 1) * delta)]))
    for i in range(B):
        if int((i + 1) * delta) < len(y_cal):
            new_values.append(np.mean(sorted_y[int(i * delta) : int((i + 1) * delta)]))
        else:
            new_values.append(np.mean(sorted_y[int(i * delta) :]))
    return intervals, new_values


def expected_calibration_error(
    prob_true: np.ndarray, prob_pred: np.ndarray, num_bins: int = 10, subsample=False
) -> Tuple[list, list, float, float, float]:
    """
    Calculates the Expected Calibration Error (ECE) for a set of predicted probabilities and true labels.

    Args:
        prob_true (numpy.ndarray): Array of true labels (0 or 1).
        prob_pred (numpy.ndarray): Array of predicted probabilities (between 0 and 1).
        num_bins (int): Number of bins to use for partitioning the predicted probabilities. Default is 10.

    Returns:
        list: Prediction average
        list: true label proportion
        float: L1 Calibration Error
        float: L2 Calibration Error
        float: max L1 Calibration error

    """
    if subsample:
        prob_true, prob_pred = balanced_subsample(y_true=prob_true, y_prob=prob_pred)
    # Sort the predicted probabilities in ascending order
    sorted_indices = np.argsort(prob_pred)
    prob_true = prob_true[sorted_indices]
    prob_pred = prob_pred[sorted_indices]

    # Partition the predicted probabilities into bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece_l1 = 0
    ece_l2 = 0
    max_l1 = []
    mean_proba = []
    mean_true = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Compute the average predicted probability and true fraction in the bin
        bin_mask = np.logical_and(prob_pred >= bin_lower, prob_pred < bin_upper)
        bin_prob_pred = prob_pred[bin_mask]
        bin_prob_true = prob_true[bin_mask]

        if len(bin_prob_true) > 0:
            avg_pred_prob = np.mean(bin_prob_pred)
            true_fraction = np.mean(bin_prob_true)

            # Compute the weighted contribution to the ECE
            bin_weight = len(bin_prob_true) / len(prob_true)
            ece_l1 += bin_weight * np.abs(true_fraction - avg_pred_prob)
            ece_l2 += bin_weight * np.square(true_fraction - avg_pred_prob)
            max_l1.append(np.abs(true_fraction - avg_pred_prob))

            # save probs
            mean_proba.append(np.mean(bin_prob_pred))
            mean_true.append(np.mean(bin_prob_true))
    return mean_proba, mean_true, ece_l1, ece_l2, np.max(max_l1)


def calibrated_mapping(x: np.ndarray, intervals: list, new_values: list) -> np.ndarray:
    bin_indices = np.digitize(x, [interval[1] for interval in intervals], right=True)
    new_values = np.array(new_values)
    return new_values[bin_indices - 1]

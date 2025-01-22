from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.utils import resample

def balanced_subsample(y_true: np.ndarray, y_prob: np.ndarray):
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
    print(f"new len: {len(y_true)}")
    return y_true, y_prob


def load_data(dataset_str: str) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_str == 'credit':
        df = pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
            header=1)

        target = 'default payment next month'
        features = [x for x in df.columns if x not in (target, 'ID')]
        X, y = df[features], df[target]

        print('Number of total samples:    {}'.format(X.shape[0]))
        print('Number of positive samples: {}'.format(y.sum()))
        # do some undersampling, to make the class more balanced
        idx_neg = X[y == 0].sample(y.sum()).index.values
        idx_pos = X[y == 1].index.values
        idx = np.concatenate([idx_neg, idx_pos])
        X = X.iloc[idx, :].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    else:
        raise ValueError("Dataset not supported or no dataset selected")
    return X, y


def bin_calibrate(y_pred: np.ndarray,
              y_cal: np.ndarray,
              B: int) -> Tuple[np.ndarray, np.ndarray]:
    ''' fixed width bin calibration with B bins
    :param y_pred: predictions
    :param y_cal: target
    :param B: number of bins
    :return: calibrated intervals and values
    '''

    sorted_y = np.asarray(y_cal)[np.argsort(y_pred)]
    scores = np.asarray(y_pred)[np.argsort(y_pred)]
    delta = int(len(y_cal) + 1) / B
    intervals = []
    new_values = []
    for i in range(B):
        if i == 0:
          intervals.append((0, scores[int((i+1)*delta)]))
        elif i == B-1:
          intervals.append((scores[int(i*delta)], 1))
        else:
          intervals.append((scores[int(i*delta)], scores[int((i+1)*delta)]))
    for i in range(B):
        if int((i+1)*delta) < len(y_cal):
          new_values.append(np.mean(sorted_y[int(i*delta):int((i+1)*delta)]))
        else:
          new_values.append(np.mean(sorted_y[int(i*delta):]))
    return intervals, new_values

def expected_calibration_error(prob_true : np.ndarray,
                               prob_pred : np.ndarray,
                               num_bins : int = 10,
                               subsample = False) -> Tuple[list, list, float, float, float]:
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


def calibrated_mapping(x : np.ndarray,
                       intervals : list,
                       new_values : list) -> np.ndarray:
    bin_indices = np.digitize(x, [interval[1] for interval in intervals], right=True)
    new_values = np.array(new_values)
    return new_values[bin_indices - 1]
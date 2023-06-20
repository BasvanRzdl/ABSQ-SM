import math
import numpy as np

def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    result = 0
    for i in range(0, len(y_true)):
        true_vec = y_true[i]
        pred_vec = y_pred[i]
        result += (1/3) * sum(abs(pred_vec[i] - true_vec[i]) for i in range(3))
    return result / len(y_true)


def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_smooth = smoothing(y_true)

    result = 0
    for i in range(0, len(y_true_smooth)):
        true_vec = y_true_smooth[i]
        pred_vec = y_pred[i]
        result += (1/3) * sum((abs(pred_vec[i] - true_vec[i]) / true_vec[i] for i in range(3)))

    return result / len(y_true_smooth)


def kullback_leibler_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_smooth = smoothing(y_pred)
    result = 0
    for i in range(0, len(y_true)):
        true_vec = y_true[i]
        pred_vec = y_pred_smooth[i]
        result += sum(true_vec[i] * np.log(true_vec[i] / (pred_vec[i])) for i in range(3))
    return result / len(y_true)


def smoothing(y_true: np.ndarray) -> np.ndarray:
    y_true_smooth = np.zeros(shape=(len(y_true), 3))
    epsilon = 1 / (2 * 3)

    for i in range(len(y_true)):
        for j in range(3):
            y_true_smooth[i, j] = (abs(epsilon + y_true[i,j]) / (epsilon*3 + sum(y_true[i])))

    return y_true_smooth

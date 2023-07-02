from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from scipy import stats
import numpy as np


def icc(pred, target):
    # Intra-class Correlation Coefficient (ICC)
    n = len(pred)
    mean_pred = np.mean(pred)
    mean_target = np.mean(target)
    ss_total = np.sum(np.square(target - mean_target)) + np.sum(
        np.square(pred - mean_pred)
    )
    ss_error = np.sum(np.square(target - pred))
    icc_val = (ss_total - ss_error) / ss_total
    return icc_val


def evaluate_continuous(pred, target):
    r = stats.pearsonr(pred, target)[0]
    mae = mean_absolute_error(pred, target)
    r2 = r2_score(pred, target)
    mse = mean_squared_error(pred, target)
    rmse = np.sqrt(mse)
    icc_val = icc(pred, target)
    # TODO: Fix this, ,explained_variance_score is not defined
    # explained_variance = explained_variance_score(pred, target)
    metrics = {
        "Pearson r": r,
        "MAE": mae,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "ICC": icc_val,
    }

    return metrics


def evaluate_binary(pred, target, threshold=0.5):
    # Convert probability predictions to binary predictions using the threshold
    binary_pred = np.where(np.array(pred) >= threshold, 1, 0)

    accuracy = accuracy_score(target, binary_pred)
    precision = precision_score(target, binary_pred)
    recall = recall_score(target, binary_pred)
    f1 = f1_score(target, binary_pred)
    roc_auc = roc_auc_score(target, pred)  # Use probability predictions for ROC AUC
    cm = confusion_matrix(target, binary_pred)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Confusion Matrix": cm,
    }

    return metrics

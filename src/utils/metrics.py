import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


def compute_metrics(y_true, y_pred):
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred,    average='macro', zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred,        average='macro', zero_division=0)),
    }


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
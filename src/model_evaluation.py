""" Functions for evaluating ResNet model"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from numpy import ndarray
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from src.resnet50_arch import ResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_test_score_list(model: ResNet, test_loader: DataLoader) -> Tuple[list, list]:
    """"""

    model.eval()
    y_test = []
    y_score = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            y_test.extend(targets.cpu().numpy())
            y_score.extend(predictions.cpu().numpy())

    return y_test, y_score


def create_roc_calc_auc(
    y_test: list, y_score: list
) -> Tuple[ndarray, ndarray, ndarray, float]:
    """"""
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=0)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


def plot_roc_auc(
    fpr: ndarray, tpr: ndarray, thresholds: ndarray, roc_auc: float
) -> plt:
    """"""
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")

    return plt

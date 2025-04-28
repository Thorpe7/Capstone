""" Functions for evaluating ResNet model"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Tuple
from numpy import ndarray
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from src.resnet_arch import ResNet

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
            logits = model(inputs)
            # _, predictions = torch.max(logits, 1)

            y_test.extend(targets.cpu().numpy())
            y_score.extend(logits.cpu().numpy())

    return y_test, y_score


def one_vs_rest_roc_calc(
    y_test: list, y_score: list, classes: tuple
) -> Tuple[ndarray, ndarray, ndarray, float]:
    """"""
    # Formatting
    num_classes = len(classes)
    y_test_one_hot = np.eye(num_classes)[y_test]
    y_score_one_hot = np.array(y_score)

    # Store values
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Iterate and calculate
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score_one_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plot_roc_auc(fpr: dict, tpr: dict, roc_auc: dict, classes: tuple) -> plt:
    """"""
    num_classes = len(classes)

    plt.figure(figsize=(12, 8))

    # Iterate through each class's fpr and tpr to plot ROC curve
    colors = cycle(["blue", "red", "green", "orange"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{classes[i]} (area = {round(roc_auc[i],2)})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc="lower right")

    return plt


def plot_loss_curve(train_loss_values: list, validation_loss_values) -> plt:
    """"""
    # Plotting the training loss
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(train_loss_values, label="Training Loss")
    plt.plot(validation_loss_values, label="Validation Loss")

    # Adding title and labels
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Adding legend
    plt.legend()

    # Show the plot
    return plt


def get_performance_metrics(
    confusion_matrix: np.ndarray, classes: tuple
) -> pd.DataFrame:
    # Initialize arrays to store recall and F1 score for each class
    recall = np.zeros(confusion_matrix.shape[0])
    f1_score = np.zeros(confusion_matrix.shape[0])

    # Compute recall and F1 score for each class
    for i in range(confusion_matrix.shape[0]):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        precision = round(TP / (TP + FP) if (TP + FP) != 0 else 0, 2)
        recall[i] = round(TP / (TP + FN) if (TP + FN) != 0 else 0, 2)
        f1_score[i] = (
            2 * (precision * recall[i]) / (precision + recall[i])
            if (precision + recall[i]) != 0
            else 0
        )

    # Create a DataFrame
    scores_df = pd.DataFrame(
        {
            "Class": [f"Class {classes[i]}" for i in range(confusion_matrix.shape[0])],
            "Recall": recall,
            "F1 Score": f1_score,
        }
    )

    return scores_df

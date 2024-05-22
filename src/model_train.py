import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import logging as log
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

log.basicConfig(level=log.INFO)
log = log.getLogger(__name__)


def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, device):
    log.info("Model training started...")
    model = model.to(device)

    # Set-up training parameters
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001
    )
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=20, mode="min")
    scheduler = MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1) 

    epoch_num = []
    train_acc_list, train_loss_list = [], []
    valid_acc_list, valid_loss_list = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Scheduler step here if using MultiStepLR
        scheduler.step()
        model.train()
        total_train_loss = 0

        # Iterate through batches
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(features)
            train_loss = loss_function(logits, labels)

            # Back prop & update model parameters
            train_loss.backward()
            optimizer.step()

            # Log training loss
            total_train_loss += train_loss.item()

        # Record training accuracy & loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        train_acc = compute_accuracy(model, train_loader, device, "Training")
        train_acc_list.append(train_acc)

        # Set to eval for validation
        model.eval()

        total_valid_loss = 0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                valid_loss = loss_function(logits, labels)
                total_valid_loss += valid_loss.item()

        # Record validation accuracy & loss
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_list.append(avg_valid_loss)
        valid_acc = compute_accuracy(model, valid_loader, device, "Validation")
        valid_acc_list.append(valid_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch number
        epoch_num.append(epoch + 1)

        # Apply scheduler if ReduceLROnPlateau is used
        # scheduler.step(avg_valid_loss)

        # Log training progress
        log.info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Valid Loss: {avg_valid_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}, "
            f"Valid Acc: {valid_acc:.2f}, "
            f"Current LR: {current_lr}"
        )

    log.info("Training Completed...")
    return (
        model,
        epoch_num,
        train_acc_list,
        valid_acc_list,
        train_loss_list,
        valid_loss_list,
    )


def test_model(test_loader, device, model, batch_size, classes):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # max returns
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(images.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        log.info(f"Accuracy of the network: {acc} %")

        for i in range(len(classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            log.info(f"Accuracy of {classes[i]}: {acc} %")


def compute_accuracy(model, data_loader, device, acc_type, model_status=None):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    if model_status == "post":
        log.info(f"{acc_type} Accuracy: {correct_pred.float()/num_examples * 100}")
    computed_accuracy = correct_pred.float() / num_examples * 100
    return round(computed_accuracy.item(), 2)


def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to("cpu"))
            all_predictions.extend(predicted_labels.to("cpu"))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    print("All targets")
    print(all_targets)
    print("all predictions")
    print(all_predictions)
    return mat


def plot_confusion_matrix(matrix: np.ndarray, classes: tuple):
    # Set the context for the plot
    sns.set_context("talk")

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )

    # Add labels and title (optional)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")

    # Show the plot
    return plt


def plot_accuracy_per_iter(epoch_num: list, accuracy_list: list, name_label: str):
    plt.plot(epoch_num, accuracy_list, label=name_label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Time")
    plt.legend()
    return plt


def compute_error_rate(model, data_loader, device, acc_type):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

        accuracy = correct_pred.float() / num_examples * 100
        error_rate = 100 - accuracy  # Calculate the error rate

    log.info(f"{acc_type} Error Rate: {error_rate}%")
    return round(error_rate.item(), 2)

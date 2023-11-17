import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import logging as log
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)
log = log.getLogger(__name__)


def train_model(
    model, train_loader, valid_loader, num_epochs, batch_size, learning_rate, device
):
    log.info("Model training started...")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , momentum=0.9

    n_total_steps = len(train_loader)
    epoch_num = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(features)
            loss = loss_function(logits, labels)

            # Back prop
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                log.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )

        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device, "Training")
            valid_acc = compute_accuracy(model, valid_loader, device, "Validation")
            epoch_num.append(epoch + 1)
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

    log.info("Training Completed...")
    return model, epoch_num, train_acc_list, valid_acc_list


def test_model(test_loader, device, model, batch_size, classes):
    model = model.to(device)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
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

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            log.info(f"Accuracy of {classes[i]}: {acc} %")


def compute_accuracy(model, data_loader, device, acc_type):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
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


def plot_accuracy_per_iter(epoch_num: list, accuracy_list: list, name_label: str):
    plt.plot(epoch_num, accuracy_list, label=name_label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Time")
    plt.legend()
    plt.show()

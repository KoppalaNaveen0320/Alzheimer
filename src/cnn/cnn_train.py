import os
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ----------------------------------
# Fix Import Path
# ----------------------------------

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from cnn_dataset import get_dataloaders
from cnn_model import AlzheimerCNN

# ----------------------------------
# Device
# ----------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {device}")

# ----------------------------------
# Hyperparameters
# ----------------------------------

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
PATIENCE = 3

# ----------------------------------
# Load Dataset
# ----------------------------------

train_loader, val_loader, test_loader, classes = (
    get_dataloaders(
        batch_size=BATCH_SIZE
    )
)

print("\nClasses:")
print(classes)

# ----------------------------------
# Model
# ----------------------------------

model = AlzheimerCNN(
    num_classes=4
).to(device)

# ----------------------------------
# Loss Function
# ----------------------------------

criterion = nn.CrossEntropyLoss()

# ----------------------------------
# Optimizer
# ----------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ----------------------------------
# Output Directories
# ----------------------------------

os.makedirs(
    "outputs/cnn_models",
    exist_ok=True
)

os.makedirs(
    "outputs/cnn_results",
    exist_ok=True
)

# ----------------------------------
# Training History
# ----------------------------------

train_losses = []
val_accuracies = []

best_acc = 0.0
early_stop_counter = 0

# ----------------------------------
# Training Loop
# ----------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    start = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # -------------------------
        # Progress Monitor
        # -------------------------

        if batch_idx > 0 and batch_idx % 100 == 0:

            elapsed = time.time() - start

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"{elapsed/100:.2f} sec/batch"
            )

            start = time.time()

    # ----------------------------------
    # Validation
    # ----------------------------------

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            preds.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )
    avg_loss = (
        running_loss /
        len(train_loader)
    )

    train_losses.append(
        avg_loss
    )
    
    val_acc = accuracy_score(
        targets,
        preds
    )

    val_accuracies.append(
        val_acc
    )

    scheduler.step(
        val_acc
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss={avg_loss:.4f} "
        f"ValAcc={val_acc:.4f}"
    )

    # ----------------------------------
    # Save Best Model
    # ----------------------------------

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "outputs/cnn_models/best_cnn_model.pth"
        )

        early_stop_counter = 0

        print(
            "Best Model Saved"
        )

    else:

        early_stop_counter += 1

    # ----------------------------------
    # Early Stopping
    # ----------------------------------

    if early_stop_counter >= PATIENCE:

        print(
            "\nEarly Stopping Triggered"
        )

        break

# ----------------------------------
# Load Best Model
# ----------------------------------

print(
    "\nLoading Best Model..."
)

model.load_state_dict(
    torch.load(
        "outputs/cnn_models/best_cnn_model.pth",
        map_location=device
    )
)

model.eval()

# ----------------------------------
# Test Evaluation
# ----------------------------------

preds = []
targets = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        preds.extend(
            predicted.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

# ----------------------------------
# Metrics
# ----------------------------------

test_acc = accuracy_score(
    targets,
    preds
)

print("\n==========================")
print("FINAL TEST RESULTS")
print("==========================")

print(
    f"Test Accuracy: {test_acc:.4f}"
)

print(
    "\nClassification Report\n"
)

print(
    classification_report(
        targets,
        preds,
        target_names=classes
    )
)

# ----------------------------------
# Confusion Matrix
# ----------------------------------

cm = confusion_matrix(
    targets,
    preds
)

print("\nConfusion Matrix")
print(cm)

# ----------------------------------
# Loss Graph
# ----------------------------------

plt.figure(
    figsize=(8,5)
)

plt.plot(
    train_losses
)

plt.title(
    "Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.savefig(
    "outputs/cnn_results/loss_curve.png"
)

plt.close()

# ----------------------------------
# Accuracy Graph
# ----------------------------------

plt.figure(
    figsize=(8,5)
)

plt.plot(
    val_accuracies
)

plt.title(
    "Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    "outputs/cnn_results/accuracy_curve.png"
)

plt.close()

print(
    "\nSaved:"
)

print(
    "outputs/cnn_models/best_cnn_model.pth"
)

print(
    "outputs/cnn_results/loss_curve.png"
)

print(
    "outputs/cnn_results/accuracy_curve.png"
)
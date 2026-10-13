import os
import time
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from validator_dataset import get_dataloaders
from validator_model import MRIValidator

# ----------------------------------
# Device
# ----------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {device}")

# ----------------------------------
# Dataset
# ----------------------------------

train_loader, val_loader, test_loader, classes = get_dataloaders(
    batch_size=32
)

print("\nClasses Found:")
print(classes)

# ----------------------------------
# Model
# ----------------------------------

model = MRIValidator().to(device)

# ----------------------------------
# Loss Function
# ----------------------------------

criterion = nn.CrossEntropyLoss()

# ----------------------------------
# Optimizer
# ----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ----------------------------------
# Output Folder
# ----------------------------------

os.makedirs(
    "outputs/mri_validator",
    exist_ok=True
)

# ----------------------------------
# Training Config
# ----------------------------------

EPOCHS = 5

best_acc = 0.0

# ----------------------------------
# Training Loop
# ----------------------------------

for epoch in range(EPOCHS):

    epoch_start_time = time.time()

    model.train()

    running_loss = 0.0

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

        # Progress every 20 batches
        if batch_idx > 0 and batch_idx % 20 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)}"
            )

    # ----------------------------------
    # Validation
    # ----------------------------------

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            preds.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        targets,
        preds
    )

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_time = (
        time.time() -
        epoch_start_time
    )

    avg_batch_time = (
        epoch_time /
        len(train_loader)
    )

    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
        f" | Loss={avg_loss:.4f}"
        f" | ValAcc={acc:.4f}"
        f" | EpochTime={epoch_time:.2f}s"
        f" | BatchTime={avg_batch_time:.3f}s"
    )

    # ----------------------------------
    # Save Best Model
    # ----------------------------------

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "outputs/mri_validator/best_validator.pth"
        )

        print(
            "Best Model Saved"
        )

# ----------------------------------
# Training Complete
# ----------------------------------

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print(
    f"Best Validation Accuracy: "
    f"{best_acc:.4f}"
)

print(
    "\nModel Saved:"
)

print(
    "outputs/mri_validator/best_validator.pth"
)
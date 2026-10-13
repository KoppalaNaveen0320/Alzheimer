import os
import sys
import torch
import pandas as pd
import numpy as np

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------
# Fix Import Path
# -------------------

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        ".."
    )
)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gat_model import GATModel

# -------------------
# Config
# -------------------

BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 100
PATIENCE = 15

torch.manual_seed(42)
np.random.seed(42)

# -------------------
# Device
# -------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {device}")

# -------------------
# Load Graph Dataset
# -------------------

graphs = torch.load(
    "processed/graphs/graph_dataset.pt",
    weights_only=False
)

print(f"Total Graphs: {len(graphs)}")

# -------------------
# Load CSV Files
# -------------------

master_df = pd.read_csv(
    "processed/master_dataset.csv"
)

train_df = pd.read_csv(
    "processed/splits/train.csv"
)

val_df = pd.read_csv(
    "processed/splits/val.csv"
)

test_df = pd.read_csv(
    "processed/splits/test.csv"
)

NUM_CLASSES = int(master_df["Label"].max()) + 1

# -------------------
# Graph Mapping
# -------------------

graph_map = {}

for idx, row in master_df.iterrows():

    graph_map[
        row["Subject_ID"]
    ] = graphs[idx]

# -------------------
# Build Splits
# -------------------

train_graphs = [
    graph_map[s]
    for s in train_df["Subject_ID"]
    if s in graph_map
]

val_graphs = [
    graph_map[s]
    for s in val_df["Subject_ID"]
    if s in graph_map
]

test_graphs = [
    graph_map[s]
    for s in test_df["Subject_ID"]
    if s in graph_map
]

print(
    f"Train Graphs: {len(train_graphs)}"
)

print(
    f"Validation Graphs: {len(val_graphs)}"
)

print(
    f"Test Graphs: {len(test_graphs)}"
)

# -------------------
# Data Loaders
# -------------------

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------
# Class Weights
# -------------------

label_counts = (
    train_df["Label"]
    .value_counts()
    .sort_index()
)

weights = []

for class_id in range(NUM_CLASSES):
    class_count = label_counts.get(class_id, 0)

    if class_count == 0:
        weights.append(1.0)
    else:
        weights.append(
            len(train_df)
            /
            (
                NUM_CLASSES
                * class_count
            )
        )

class_weights = torch.tensor(
    weights,
    dtype=torch.float
).to(device)

print(
    "\nClass Weights:"
)

print(class_weights)

# -------------------
# Model
# -------------------

model = GATModel(
    input_dim=5,
    hidden_dim=64,
    num_classes=NUM_CLASSES
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )
)

criterion = (
    torch.nn.CrossEntropyLoss(
        weight=class_weights
    )
)

# -------------------
# Output Folder
# -------------------

os.makedirs(
    "outputs/models",
    exist_ok=True
)

best_val_acc = 0.0
patience_counter = 0

# -------------------
# Training
# -------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        outputs = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(
            outputs,
            batch.y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    # -------------------
    # Validation
    # -------------------

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(device)

            outputs = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            predicted = (
                outputs.argmax(
                    dim=1
                )
            )

            preds.extend(
                predicted.cpu().numpy()
            )

            labels.extend(
                batch.y.cpu().numpy()
            )

    val_acc = accuracy_score(
            labels,
            preds
        )

    val_precision = precision_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    val_recall = recall_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    val_f1 = f1_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    scheduler.step(
        val_acc
    )

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Acc: {val_acc:.4f} | "
        f"Precision: {val_precision:.4f} | "
        f"Recall: {val_recall:.4f} | "
        f"F1: {val_f1:.4f}"
    )

    # -------------------
    # Save Best Model
    # -------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        patience_counter = 0

        torch.save(
            model.state_dict(),
            "outputs/models/best_gat_model.pth"
        )

    else:

        patience_counter += 1

    # -------------------
    # Early Stopping
    # -------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly Stopping Triggered"
        )

        break

# -------------------
# Finish
# -------------------

print("\nTraining Complete")

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)

print(
    "\nModel Saved:"
)

print(
    "outputs/models/best_gat_model.pth"
)

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from torch_geometric.loader import DataLoader

# -------------------
# Fix Import Path (Problem 1 Fix)
# -------------------
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# PROJECT_ROOT ని కరెక్ట్ రూట్ డైరెక్టరీకి పాయింట్ చేయడం జరిగింది
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gat_model import GATModel

# -------------------
# Device
# -------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# -------------------
# Load Graph Dataset & DataFrames (Problem 2 Fix)
# -------------------
graphs_path = os.path.join(PROJECT_ROOT, "processed/graphs/graph_dataset.pt")
master_path = os.path.join(PROJECT_ROOT, "processed/master_dataset.csv")

# Corrected test CSV path
test_path = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "test.csv"
)

graphs = torch.load(graphs_path, weights_only=False)
master_df = pd.read_csv(master_path)
test_df = pd.read_csv(test_path)

# -------------------
# Graph Mapping
# -------------------
graph_map = {}
for idx, row in master_df.iterrows():
    graph_map[row["Subject_ID"]] = graphs[idx]

test_graphs = []
for subject_id in test_df["Subject_ID"]:
    if subject_id in graph_map:
        test_graphs.append(graph_map[subject_id])

print(f"Test Graphs Found: {len(test_graphs)}")

# -------------------
# DataLoader
# -------------------
test_loader = DataLoader(
    test_graphs,
    batch_size=16,
    shuffle=False
)

# -------------------
# Load Model (4-Classes Configured)
# -------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/models/best_gat_model.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = GATModel(
    input_dim=5,
    hidden_dim=64,
    num_classes=4
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.eval()

# -------------------
# Prediction Pipeline
# -------------------
preds = []
labels = []
probabilities = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        outputs = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probs = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)

        preds.extend(predicted.cpu().numpy())
        labels.extend(batch.y.cpu().numpy())
        # మల్టీ-క్లాస్ కోసం అన్ని ప్రాబబిలిటీస్‌ను సేవ్ చేస్తున్నాము (ముందు ముందు ROC కి పనికొస్తుంది)
        probabilities.extend(probs.cpu().numpy())

# -------------------
# Metrics Calculation (Problem 3 Fix)
# -------------------
acc = accuracy_score(labels, preds)

# 4-class కోసం average="weighted" కి మార్చడం జరిగింది
precision = precision_score(
    labels,
    preds,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    labels,
    preds,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    labels,
    preds,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(labels, preds)

# -------------------
# Display Results (Problem 4 Fix)
# -------------------
print("\n===== TEST RESULTS =====")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print("\nClassification Report:\n")
print(
    classification_report(
        labels,
        preds,
        target_names=[
            "Normal",
            "Very Mild",
            "Mild",
            "Moderate"
        ],
        zero_division=0
    )
)

# -------------------
# Save Confusion Matrix (Problem 5 Fix)
# -------------------
output_cm_dir = os.path.join(PROJECT_ROOT, "outputs/confusion_matrices")
os.makedirs(output_cm_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Normal",
        "Very Mild",
        "Mild",
        "Moderate"
    ],
    yticklabels=[
        "Normal",
        "Very Mild",
        "Mild",
        "Moderate"
    ]
)
plt.title("4-Class Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

cm_save_path = os.path.join(output_cm_dir, "confusion_matrix.png")
plt.savefig(cm_save_path, dpi=300)
plt.close()

print(f"\nConfusion Matrix Saved: {cm_save_path}")

# -------------------
# Save ROC Curve (Problem 6 Fix)
# -------------------
# Binary ROC Curve సెక్షన్‌ను మల్టీ-క్లాస్ ఎర్రర్స్ రాకుండా ఉండటానికి పూర్తిగా కామెంట్ చేయడం జరిగింది.
# Later, multiclass ROC can be implemented here if required.
# 
# fpr, tpr, _ = roc_curve(labels, probabilities)
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.tight_layout()
# roc_save_path = os.path.join(output_cm_dir, "roc_curve.png")
# plt.savefig(roc_save_path, dpi=300)
# plt.close()
# print(f"ROC Curve Saved: {roc_save_path}")
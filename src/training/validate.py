import os
import sys
import torch
import pandas as pd

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gat_model import GATModel

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

graphs = torch.load(
    "processed/graphs/graph_dataset.pt",
    weights_only=False
)

master_df = pd.read_csv(
    "processed/master_dataset.csv"
)

val_df = pd.read_csv(
    "processed/splits/val.csv"
)

graph_map = {}

for idx, row in master_df.iterrows():
    graph_map[row["Subject_ID"]] = graphs[idx]

val_graphs = [
    graph_map[subject_id]
    for subject_id in val_df["Subject_ID"]
    if subject_id in graph_map
]

val_loader = DataLoader(
    val_graphs,
    batch_size=16,
    shuffle=False
)

state_dict = torch.load(
    "outputs/models/best_gat_model.pth",
    map_location=device
)

num_classes = state_dict["classifier.fc2.weight"].shape[0]

model = GATModel(
    input_dim=5,
    hidden_dim=64,
    num_classes=num_classes
).to(device)

model.load_state_dict(state_dict)
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

        predicted = outputs.argmax(dim=1)

        preds.extend(
            predicted.cpu().numpy()
        )

        labels.extend(
            batch.y.cpu().numpy()
        )

acc = accuracy_score(
    labels,
    preds
)

print("\n===== VALIDATION RESULT =====")
print(f"Validation Accuracy: {acc:.4f}")

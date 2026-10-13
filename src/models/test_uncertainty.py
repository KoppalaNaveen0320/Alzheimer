import os
import sys
import torch

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gat_model import GATModel
from models.uncertainty_head import (
    UncertaintyEstimator
)

graphs = torch.load(
    "processed/graphs/graph_dataset.pt",
    weights_only=False
)

graph = graphs[0]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
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
graph.batch = torch.zeros(
    graph.num_nodes,
    dtype=torch.long
)

estimator = UncertaintyEstimator(
    model,
    device=device,
    num_samples=50
)

result = estimator.predict_with_uncertainty(
    graph
)

print("\n===== UNCERTAINTY RESULT =====")

print(
    "Predicted Class:",
    result["predicted_class"]
)

print(
    "Confidence:",
    round(
        result["confidence"],
        4
    )
)

print(
    "Uncertainty:",
    round(
        result["uncertainty"],
        4
    )
)

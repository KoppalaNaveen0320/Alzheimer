import os
import sys
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# ---------------------------------
# Project Paths
# ---------------------------------
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ---------------------------------
# Import Model
# ---------------------------------
from models.gat_model import GATModel

# ---------------------------------
# Device
# ---------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# ---------------------------------
# Model Path
# ---------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/models/best_gat_model.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

# ---------------------------------
# Load Model
# ---------------------------------
state_dict = torch.load(MODEL_PATH, map_location=device)
classifier_weight = state_dict.get("classifier.fc2.weight")

if classifier_weight is None:
    raise ValueError(
        "Checkpoint is missing classifier.fc2.weight"
    )

num_classes = classifier_weight.shape[0]

model = GATModel(
    input_dim=5,
    hidden_dim=64,
    num_classes=num_classes
).to(device)

model.load_state_dict(state_dict)
model.eval()

print("Model Loaded Successfully")

# ---------------------------------
# Required ROI Features
# ---------------------------------
FEATURE_COLUMNS = [
    "Mean",
    "Std",
    "Min",
    "Max",
    "Volume"
]

CORRELATION_THRESHOLD = 0.30

# ---------------------------------
# Build Graph
# ---------------------------------
def build_graph(roi_df):
    for col in FEATURE_COLUMNS:
        if col not in roi_df.columns:
            raise ValueError(
                f"Missing ROI Column: {col}"
            )

    features = roi_df[FEATURE_COLUMNS].values
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    num_nodes = len(features)
    edge_index = []

    corr_matrix = np.corrcoef(features)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            corr_value = corr_matrix[i][j]

            if np.isnan(corr_value):
                continue

            if abs(corr_value) >= CORRELATION_THRESHOLD:
                edge_index.append([i, j])

    if len(edge_index) == 0:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    x = torch.tensor(
        features,
        dtype=torch.float
    )

    graph = Data(x=x, edge_index=edge_index)
    
    # Batch tensor for single graph prediction
    graph.batch = torch.zeros(
        graph.num_nodes,
        dtype=torch.long
    )

    return graph


def predict_roi_dataframe(roi_df):
    graph = build_graph(roi_df)
    graph = graph.to(device)

    with torch.no_grad():
        outputs = model(
            graph.x,
            graph.edge_index,
            graph.batch
        )

        probs = torch.softmax(outputs, dim=1)

        prediction = int(
            torch.argmax(probs, dim=1).item()
        )

        confidence = float(
            probs.max().item()
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "probabilities": probs.cpu().numpy()[0].tolist()
    }

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_roi_csv(roi_csv_path):
    if not os.path.exists(roi_csv_path):
        raise FileNotFoundError(roi_csv_path)

    roi_df = pd.read_csv(roi_csv_path)
    return predict_roi_dataframe(roi_df)

# ---------------------------------
# Label Converter
# ---------------------------------
def get_prediction_label(prediction):

    labels = {
        0: "NORMAL",
        1: "VERY MILD DEMENTIA",
        2: "MILD DEMENTIA",
        3: "MODERATE DEMENTIA",
        4: "SEVERE DEMENTIA"
    }

    return labels.get(
        prediction,
        "UNKNOWN"
    )

# ---------------------------------
# Human Explanation
# ---------------------------------

def explain_result(
    prediction,
    confidence
):

    if confidence < 0.60:

        return f"""
Prediction Confidence is Low

Confidence: {confidence:.2%}

The model is not sufficiently
certain about this prediction.

Additional MRI analysis and
clinical evaluation are
recommended.
"""

    if prediction > 0:

        return f"""
Possible Alzheimer's related
patterns detected:
{get_prediction_label(prediction)}

Confidence: {confidence:.2%}

The extracted brain ROI features
show similarity with Alzheimer's
cases present in the training
dataset.

This result is intended only for
screening purposes and should not
be considered a medical diagnosis.
"""

    return f"""
No strong Alzheimer's related
patterns detected.

Confidence: {confidence:.2%}

The extracted brain ROI features
appear closer to normal patterns
observed in the training dataset.

This result is intended only for
screening purposes and should not
be considered a medical diagnosis.
"""

# ---------------------------------
# Example Run
# ---------------------------------
if __name__ == "__main__":
    SAMPLE_FILE = os.path.join(
        PROJECT_ROOT, 
        "processed/roi_features/mwrc1OAS1_0001_MR1_mpr_anon_fslswapdim_bet.csv"
    )

    print("\nRunning Sample Prediction...")
    
    try:
        result = predict_roi_csv(SAMPLE_FILE)
        label = get_prediction_label(result["prediction"])

        print("\n===== RESULT =====")
        print(f"Prediction  : {label}")
        print(f"Confidence  : {result['confidence']:.4f}")
        print(f"Uncertainty : {result['uncertainty']:.4f}")

        print("\n===== EXPLANATION =====")
        print(explain_result(result["prediction"], result["confidence"]))
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

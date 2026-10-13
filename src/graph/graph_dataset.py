import os
import torch
import pandas as pd
import numpy as np

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# Configuration
# -----------------------------------

MASTER_DATASET = (
    "processed/master_dataset.csv"
)

GRAPH_SAVE_PATH = (
    "processed/graphs/graph_dataset.pt"
)

CORRELATION_THRESHOLD = 0.30

FEATURE_COLUMNS = [
    "Mean",
    "Std",
    "Min",
    "Max",
    "Volume"
]

# -----------------------------------
# Load Dataset
# -----------------------------------

dataset = pd.read_csv(
    MASTER_DATASET
)

graphs = []

print(
    f"\nLoaded Dataset: {len(dataset)} samples"
)

# -----------------------------------
# Build Graphs
# -----------------------------------

for idx, row in dataset.iterrows():

    try:

        roi_file = row["ROI_File"]

        label = int(
            row["Label"]
        )

        if not os.path.exists(
            roi_file
        ):

            print(
                f"Missing ROI File: {roi_file}"
            )

            continue

        roi_df = pd.read_csv(
            roi_file
        )

        # -----------------------------
        # Feature Validation
        # -----------------------------

        missing_columns = []

        for col in FEATURE_COLUMNS:

            if col not in roi_df.columns:

                missing_columns.append(
                    col
                )

        if len(
            missing_columns
        ) > 0:

            print(
                f"Skipping {roi_file}"
            )

            print(
                f"Missing Columns: {missing_columns}"
            )

            continue

        # -----------------------------
        # Node Features
        # -----------------------------

        node_features = roi_df[
            FEATURE_COLUMNS
        ].values

        scaler = StandardScaler()

        node_features = (
            scaler.fit_transform(
                node_features
            )
        )

        node_features = np.nan_to_num(
            node_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        num_nodes = len(
            node_features
        )

        # -----------------------------
        # Correlation Matrix
        # -----------------------------

        corr_matrix = np.corrcoef(
            node_features
        )

        edge_index = []

        for i in range(
            num_nodes
        ):

            for j in range(
                num_nodes
            ):

                if i == j:
                    continue

                corr_value = (
                    corr_matrix[i][j]
                )

                if np.isnan(
                    corr_value
                ):
                    continue

                if abs(
                    corr_value
                ) >= CORRELATION_THRESHOLD:

                    edge_index.append(
                        [i, j]
                    )

        # -----------------------------
        # Graph Fallback
        # -----------------------------

        if len(
            edge_index
        ) == 0:

            print(
                f"Fallback Graph Used:"
                f" {roi_file}"
            )

            for i in range(
                num_nodes
            ):

                for j in range(
                    num_nodes
                ):

                    if i != j:

                        edge_index.append(
                            [i, j]
                        )

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        # -----------------------------
        # Tensor Conversion
        # -----------------------------

        x = torch.tensor(
            node_features,
            dtype=torch.float
        )

        y = torch.tensor(
            [label],
            dtype=torch.long
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        graphs.append(
            graph
        )

        # -----------------------------
        # Progress
        # -----------------------------

        if (
            idx + 1
        ) % 50 == 0:

            print(
                f"Processed "
                f"{idx+1}/"
                f"{len(dataset)}"
            )

    except Exception as e:

        print(
            f"\nError Processing:"
        )

        print(
            roi_file
        )

        print(
            str(e)
        )

# -----------------------------------
# Dataset Statistics
# -----------------------------------

print(
    "\n" + "=" * 60
)

print(
    "GRAPH DATASET SUMMARY"
)

print(
    "=" * 60
)

print(
    f"\nTotal Graphs:"
    f" {len(graphs)}"
)

if len(graphs) > 0:

    first_graph = graphs[0]

    print(
        f"Feature Dimension:"
        f" {first_graph.x.shape[1]}"
    )

    print(
        f"Nodes Per Graph:"
        f" {first_graph.x.shape[0]}"
    )

    print(
        f"Edges In First Graph:"
        f" {first_graph.edge_index.shape[1]}"
    )

    labels = [
        int(g.y.item())
        for g in graphs
    ]

    print(
        "\nUnique Labels:"
    )

    print(
        sorted(
            list(
                set(labels)
            )
        )
    )

# -----------------------------------
# Save Graph Dataset
# -----------------------------------

os.makedirs(
    "processed/graphs",
    exist_ok=True
)

torch.save(
    graphs,
    GRAPH_SAVE_PATH
)

print(
    "\nSaved:"
)

print(
    GRAPH_SAVE_PATH
)

print(
    "\nGraph Dataset Creation Complete"
)
import pandas as pd
from collections import Counter
import torch
import os

print("=" * 60)
print("ALZHEIMER DATASET INSPECTION")
print("=" * 60)

# --------------------------------
# Check Master Dataset
# --------------------------------

master_path = "processed/master_dataset.csv"

if os.path.exists(master_path):

    df = pd.read_csv(master_path)

    print("\nMASTER DATASET FOUND")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    # Check possible severity columns

    possible_cols = [
        "CDR",
        "cdr",
        "Group",
        "group",
        "Diagnosis",
        "diagnosis",
        "Severity",
        "severity",
        "Label"
    ]

    print("\nPossible Label Columns:")

    found = False

    for col in possible_cols:

        if col in df.columns:

            found = True

            print(f"\nColumn: {col}")

            print("Unique Values:")

            print(
                df[col]
                .value_counts(dropna=False)
            )

    if not found:

        print(
            "No severity-related columns found."
        )

else:

    print(
        f"\nMaster Dataset Not Found:\n{master_path}"
    )

# --------------------------------
# Check Graph Dataset Labels
# --------------------------------

graph_path = (
    "processed/graphs/graph_dataset.pt"
)

if os.path.exists(graph_path):

    graphs = torch.load(
        graph_path,
        weights_only=False
    )

    labels = []

    for g in graphs:

        labels.append(
            int(g.y)
        )

    print("\n" + "=" * 60)
    print("GRAPH LABEL DISTRIBUTION")
    print("=" * 60)

    label_counts = Counter(labels)

    print("\nLabel Counts:")
    print(label_counts)

    print("\nUnique Labels:")
    print(sorted(set(labels)))

    print(
        f"\nTotal Graphs: {len(graphs)}"
    )

else:

    print(
        f"\nGraph Dataset Not Found:\n{graph_path}"
    )

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
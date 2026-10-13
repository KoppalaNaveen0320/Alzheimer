import pandas as pd
import torch


def load_split_graphs():

    graphs = torch.load(
        "processed/graphs/graph_dataset.pt",
        weights_only=False
    )

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

    graph_map = {}

    for idx, row in master_df.iterrows():

        graph_map[
            row["Subject_ID"]
        ] = graphs[idx]

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

    return (
        train_graphs,
        val_graphs,
        test_graphs
    )
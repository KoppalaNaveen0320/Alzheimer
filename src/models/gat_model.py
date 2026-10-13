import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    BatchNorm
)

from models.classifier import Classifier


class GATModel(torch.nn.Module):

    def __init__(
        self,
        input_dim=5,
        hidden_dim=64,
        num_classes=4
    ):
        super().__init__()

        # --------------------------------
        # GAT Layer 1
        # --------------------------------

        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=4,
            dropout=0.30
        )

        self.bn1 = BatchNorm(
            hidden_dim * 4
        )

        # --------------------------------
        # GAT Layer 2
        # --------------------------------

        self.gat2 = GATConv(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim,
            heads=2,
            dropout=0.30
        )

        self.bn2 = BatchNorm(
            hidden_dim * 2
        )

        # --------------------------------
        # GAT Layer 3
        # --------------------------------

        self.gat3 = GATConv(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            heads=1,
            dropout=0.30
        )

        self.bn3 = BatchNorm(
            hidden_dim
        )

        # --------------------------------
        # Dropout
        # --------------------------------

        self.dropout = torch.nn.Dropout(
            p=0.40
        )

        # --------------------------------
        # Classifier
        # --------------------------------

        self.classifier = Classifier(
            input_dim=hidden_dim,
            num_classes=4
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        # --------------------------------
        # GAT Block 1
        # --------------------------------

        x = self.gat1(
            x,
            edge_index
        )

        x = self.bn1(x)

        x = F.elu(x)

        x = self.dropout(x)

        # --------------------------------
        # GAT Block 2
        # --------------------------------

        x = self.gat2(
            x,
            edge_index
        )

        x = self.bn2(x)

        x = F.elu(x)

        x = self.dropout(x)

        # --------------------------------
        # GAT Block 3
        # --------------------------------

        x = self.gat3(
            x,
            edge_index
        )

        x = self.bn3(x)

        x = F.elu(x)

        # --------------------------------
        # Graph Pooling
        # --------------------------------

        x = global_mean_pool(
            x,
            batch
        )

        x = self.dropout(x)

        # --------------------------------
        # Classification
        # --------------------------------

        logits = self.classifier(x)

        return logits

    def predict(
        self,
        x,
        edge_index,
        batch
    ):

        logits = self.forward(
            x,
            edge_index,
            batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        )

        confidence = torch.max(
            probabilities,
            dim=1
        )[0]

        return (
            prediction,
            confidence,
            probabilities
        )
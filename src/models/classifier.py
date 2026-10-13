import torch.nn as nn


class Classifier(nn.Module):

    def __init__(
        self,
        input_dim=32,
        num_classes=2
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            64
        )

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x
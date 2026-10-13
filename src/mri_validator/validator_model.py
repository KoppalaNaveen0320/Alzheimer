import torch.nn as nn
from torchvision.models import mobilenet_v2

class MRIValidator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = mobilenet_v2(weights="DEFAULT")

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            2
        )

    def forward(self, x):

        return self.model(x)
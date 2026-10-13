import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class AlzheimerCNN(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):
        super().__init__()

        # ----------------------------------
        # Load Pretrained EfficientNet-B0
        # ----------------------------------

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # ----------------------------------
        # Replace Final Layer
        # ----------------------------------

        in_features = (
            self.backbone.classifier[1]
            .in_features
        )

        self.backbone.classifier = nn.Sequential(

            nn.Dropout(
                p=0.4
            ),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):
        return self.backbone(x)


# ----------------------------------
# Debug Test
# ----------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AlzheimerCNN(
        num_classes=4
    ).to(device)

    print(model)

    sample = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    output = model(sample)

    print(
        "\nOutput Shape:"
    )

    print(
        output.shape
    )
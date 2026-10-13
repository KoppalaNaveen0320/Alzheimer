import os
import torch

from PIL import Image
from torchvision import transforms

from validator_model import MRIValidator

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "mri_validator",
    "best_validator.pth"
)

print("MODEL_PATH =", MODEL_PATH)
print("FILE_EXISTS =", os.path.exists(MODEL_PATH))

CLASS_NAMES = [
    "MRI",
    "NON_MRI"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = MRIValidator().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

def validate_mri(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(
        image
    ).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence = probs.max().item()

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

    label = CLASS_NAMES[pred]

    return {
        "is_mri": label == "MRI",
        "label": label,
        "confidence": confidence
    }

if __name__ == "__main__":

    path = input(
        "Enter Image Path: "
    )

    result = validate_mri(path)

    print("\nResult")
    print(result)
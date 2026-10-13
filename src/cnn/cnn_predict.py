import os
import sys
import torch

from PIL import Image
from torchvision import transforms

# ----------------------------------
# Fix Import Path
# ----------------------------------

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from cnn_model import AlzheimerCNN

# ----------------------------------
# Device
# ----------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device: {device}")

# ----------------------------------
# Class Names
# ----------------------------------

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

# ----------------------------------
# Project Root
# ----------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

# ----------------------------------
# Model Path
# ----------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "cnn_models",
    "best_cnn_model.pth"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model Not Found:\n{MODEL_PATH}"
    )

# ----------------------------------
# Load Model
# ----------------------------------

model = AlzheimerCNN(
    num_classes=4
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("CNN Model Loaded Successfully")

# ----------------------------------
# Image Transform
# ----------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------
# Prediction Function
# ----------------------------------

def predict_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            image_path
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = float(
            probabilities.max().item()
        )

    return {

        "prediction":
        prediction,

        "label":
        CLASS_NAMES[prediction],

        "confidence":
        confidence,

        "uncertainty":
        1.0 - confidence,

        "probabilities":
        probabilities.cpu().numpy()[0]
    }

# ----------------------------------
# Confidence Threshold
# ----------------------------------

def get_prediction_label(
    label,
    confidence
):

    if confidence < 0.60:

        return "UNCERTAIN"

    return label

# ----------------------------------
# Disease Details
# ----------------------------------

def get_disease_details(label):

    details = {

        "MildDemented": {
            "stage": "Mild Demented Alzheimer's Disease",
            "risk": "Moderate",
            "description":
            "The MRI scan indicates patterns associated with the early stage of Alzheimer's disease.",

            "symptoms": [
                "Mild memory loss",
                "Difficulty remembering recent events",
                "Occasional confusion",
                "Reduced concentration"
            ],

            "recommendations": [
                "Consult a neurologist",
                "Monitor memory changes regularly",
                "Perform cognitive assessments",
                "Follow a healthy lifestyle"
            ]
        },

        "ModerateDemented": {
            "stage": "Moderate Alzheimer's Disease",
            "risk": "High",
            "description":
            "The MRI scan indicates moderate progression of Alzheimer's disease.",

            "symptoms": [
                "Significant memory impairment",
                "Difficulty recognizing people",
                "Confusion about time and place",
                "Problems with daily activities"
            ],

            "recommendations": [
                "Immediate neurological consultation",
                "Comprehensive clinical evaluation",
                "Caregiver support may be required"
            ]
        },

        "VeryMildDemented": {
            "stage": "Very Mild Demented",
            "risk": "Low to Moderate",
            "description":
            "Very early signs of cognitive decline may be present.",

            "symptoms": [
                "Minor forgetfulness",
                "Slight difficulty planning tasks",
                "Mild attention problems"
            ],

            "recommendations": [
                "Regular follow-up",
                "Brain exercises",
                "Healthy diet and sleep"
            ]
        },

        "NonDemented": {
            "stage": "Non Demented",
            "risk": "Low",
            "description":
            "No significant MRI patterns associated with Alzheimer's disease were detected.",

            "symptoms": [
                "No major dementia-related abnormalities detected"
            ],

            "recommendations": [
                "Maintain healthy lifestyle",
                "Regular medical checkups",
                "Continue cognitive activities"
            ]
        }
    }

    return details.get(label, None)

# ----------------------------------
# Human Explanation
# ----------------------------------

def explain_prediction(
    label,
    confidence
):

    if label == "UNCERTAIN":

        return f"""
Prediction : UNCERTAIN

Confidence : {confidence:.2%}

The model could not make a
reliable prediction for this image.

Possible reasons:

• Image quality is poor
• Different from training dataset
• Brain patterns are unclear

Additional MRI scans and
clinical evaluation are
recommended.

This is not a medical diagnosis.
"""

    return f"""
Prediction : {label}

Confidence : {confidence:.2%}

The uploaded image shows
patterns similar to the
predicted Alzheimer category.

This result is generated using
the trained CNN Alzheimer model.

This should only be used as an
AI screening result.

Please consult a neurologist
for professional diagnosis.
"""

# ----------------------------------
# Display Probabilities
# ----------------------------------

def print_probabilities(
    probabilities
):

    print("\nClass Probabilities")

    print(
        "-----------------------------"
    )

    for cls, prob in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"{cls:<20} "
            f"{prob:.4f}"
        )

# ----------------------------------
# Example Run
# ----------------------------------

if __name__ == "__main__":

    SAMPLE_IMAGE = input(
        "\nEnter Image Path:\n"
    ).strip()

    result = predict_image(
        SAMPLE_IMAGE
    )

    final_label = get_prediction_label(
        result["label"],
        result["confidence"]
    )

    print(
        "\n=============================="
    )

    print(
        "CNN PREDICTION RESULT"
    )

    print(
        "=============================="
    )

    print(
        f"Prediction  : "
        f"{final_label}"
    )

    print(
        f"Confidence  : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Uncertainty : "
        f"{result['uncertainty']:.4f}"
    )

    print_probabilities(
        result["probabilities"]
    )

    print(
        explain_prediction(
            final_label,
            result["confidence"]
        )
    )
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


# ----------------------------------
# Dataset Path
# ----------------------------------

DATASET_DIR = r"D:\Alzheimer_GNN_Project\data\cnn_dataset"

# ----------------------------------
# Image Transformations
# ----------------------------------

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------------
# Create DataLoaders
# ----------------------------------

def get_dataloaders(
    batch_size=32
):

    dataset = datasets.ImageFolder(
        root=DATASET_DIR,
        transform=train_transforms
    )

    print("\nClasses Found:")
    print(dataset.classes)

    print("\nClass Mapping:")
    print(dataset.class_to_idx)

    total_size = len(dataset)

    train_size = int(
        total_size * 0.70
    )

    val_size = int(
        total_size * 0.15
    )

    test_size = (
        total_size
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    val_dataset.dataset.transform = test_transforms
    test_dataset.dataset.transform = test_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset.classes
    )


# ----------------------------------
# Debug
# ----------------------------------

if __name__ == "__main__":

    train_loader, val_loader, test_loader, classes = (
        get_dataloaders()
    )

    print("\nDataset Summary")
    print("---------------------")

    print(
        f"Train Batches: {len(train_loader)}"
    )

    print(
        f"Validation Batches: {len(val_loader)}"
    )

    print(
        f"Test Batches: {len(test_loader)}"
    )

    print(
        f"Classes: {classes}"
    )
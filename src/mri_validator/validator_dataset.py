import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATASET_PATH = r"D:\Alzheimer_GNN_Project\data\mri_validator"

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_dataloaders(batch_size=32):

    dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset.classes
    )

if __name__ == "__main__":

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("\nClasses Found:")
    print(classes)

    print("\nTrain Batches:", len(train_loader))
    print("Validation Batches:", len(val_loader))
    print("Test Batches:", len(test_loader))
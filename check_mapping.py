from torchvision.datasets import ImageFolder

dataset = ImageFolder(
    r"D:\Alzheimer_GNN_Project\data\mri_validator"
)

print(dataset.class_to_idx)
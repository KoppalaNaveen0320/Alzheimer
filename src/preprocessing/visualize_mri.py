import os
import nibabel as nib
import matplotlib.pyplot as plt

DATASET_PATH = r"D:\Alzheimer_GNN_Project\data\raw\oasis1"

def visualize_first_mri():

    for subject in os.listdir(DATASET_PATH):

        subject_path = os.path.join(DATASET_PATH, subject)

        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):

            if file.endswith(".nii"):

                nii_path = os.path.join(subject_path, file)

                img = nib.load(nii_path)

                data = img.get_fdata()

                slice_idx = data.shape[2] // 2

                plt.figure(figsize=(6,6))

                plt.imshow(
                    data[:, :, slice_idx],
                    cmap="gray"
                )

                plt.title(
                    f"{subject} | Slice {slice_idx}"
                )

                plt.axis("off")
                plt.show()

                return


if __name__ == "__main__":
    visualize_first_mri()
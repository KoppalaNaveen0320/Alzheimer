import os
import glob
import nibabel as nib
import numpy as np

INPUT_DIR = "data/raw/oasis1"
OUTPUT_DIR = "processed/normalized_mri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mri_files = glob.glob(
    os.path.join(INPUT_DIR, "**", "*.nii.gz"),
    recursive=True
)

print(f"Found {len(mri_files)} MRI files")


def normalize_mri(volume):

    volume = volume.astype(np.float32)

    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val == 0:
        return volume

    volume = (volume - min_val) / (max_val - min_val)

    return volume


for idx, file_path in enumerate(mri_files):

    try:

        img = nib.load(file_path)

        data = img.get_fdata()

        normalized = normalize_mri(data)

        filename = os.path.basename(file_path)

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        new_img = nib.Nifti1Image(
            normalized,
            img.affine,
            img.header
        )

        nib.save(new_img, output_path)

        print(
            f"[{idx+1}/{len(mri_files)}] Saved -> {filename}"
        )

    except Exception as e:

        print(f"Error: {file_path}")
        print(e)

print("\nNormalization Completed")
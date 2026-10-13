import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

def load_mri(mri_path):
    img = nib.load(mri_path)
    return img, img.get_fdata()

def normalize_mri(volume):
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (
        volume.max() - volume.min() + 1e-8
    )
    return volume

def resize_mri(volume, target_shape=(128,128,128)):
    factors = [
        target_shape[i] / volume.shape[i]
        for i in range(3)
    ]
    return zoom(volume, factors)

def create_brain_mask(volume):
    threshold = np.mean(volume)
    return (volume > threshold).astype(np.float32)

def preprocess_mri(mri_path):
    img, volume = load_mri(mri_path)

    volume = normalize_mri(volume)

    volume = resize_mri(volume)

    mask = create_brain_mask(volume)

    processed = volume * mask

    os.makedirs(
        "processed/temp_mri",
        exist_ok=True
    )

    save_path = (
        "processed/temp_mri/"
        "processed_scan.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            processed,
            img.affine
        ),
        save_path
    )

    return save_path
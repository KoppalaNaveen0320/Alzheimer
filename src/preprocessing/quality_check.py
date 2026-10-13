import glob
import nibabel as nib
import numpy as np

files = glob.glob(
    "processed/normalized_mri/*.nii.gz"
)

print(f"Found {len(files)} normalized MRI files")

sample = files[0]

img = nib.load(sample)

data = img.get_fdata()

print("\n===== MRI Quality Report =====")

print("Shape :", data.shape)
print("Mean  :", np.mean(data))
print("Std   :", np.std(data))
print("Min   :", np.min(data))
print("Max   :", np.max(data))

print("\nNaN Check:")

if np.isnan(data).any():
    print("Contains NaN values")
else:
    print("No NaN values found")
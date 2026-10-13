import glob
import nibabel as nib
import numpy as np

# Search for .nii.gz files
mri_files = glob.glob(
    "data/raw/oasis1/**/*.nii.gz",
    recursive=True
)

if len(mri_files) == 0:
    print("No MRI file found")
    exit()

print(f"Found {len(mri_files)} MRI files")

sample_file = mri_files[0]

print("\nLoading:")
print(sample_file)

img = nib.load(sample_file)

data = img.get_fdata()

print("\nMRI Loaded Successfully")
print("Shape:", data.shape)
print("Datatype:", data.dtype)
print("Min:", np.min(data))
print("Max:", np.max(data))
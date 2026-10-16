from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Patient ID
patient_id = "BraTS-GLI-00000-000"

# Path to the MRI files 
data_dir = Path(__file__).resolve().parent / "data" / patient_id

# Load the FLAIR MRI scan
mri_path = data_dir / f"{patient_id}-t2f.nii.gz"
mri = nib.load(str(mri_path))

# Convert MRI to a NumPy array
mri_data = mri.get_fdata()

# Find the tumor segmentation file
seg_path = data_dir / f"{patient_id}-seg.nii.gz"

# Load the segmentation mask
seg = nib.load(str(seg_path))

# Convert it to a NumPy array
seg_data = seg.get_fdata()

# Print the tumor labels
print("Segmentation dimensions:", seg_data.shape)
print("Unique tumor labels:", np.unique(seg_data))

# Display information
print("MRI loaded successfully!")
print("MRI dimensions:", mri_data.shape)
print("Data type:", mri_data.dtype)
print("Intensity range:", np.min(mri_data), np.max(mri_data))

# Select the middle slice of the brain
slice_index = 90

# Extract a 2d slice from the 3D MRI
brain_slice = mri_data[:, :, slice_index]

# Display the MRI
plt.imshow(brain_slice.T, cmap="gray", origin="lower")
plt.title(f"Brain MRI - Slice {slice_index}")
plt.axis("off")
plt.show()


# Use the same slice as the original MRI
tumor_slice = seg_data[:, :, slice_index]

# Define segmentation colors:
# 0 = transparent background
# 1 = red
# 2 = yellow
# 3 = blue
colors = ListedColormap([
    "black",
    "red",
    "yellow",
    "blue"
])

# Display the MRI
plt.figure(figsize=(8, 8))
plt.imshow(brain_slice.T, cmap="gray", origin="lower")

# Overlay tumor labels
plt.imshow(
    np.ma.masked_where(tumor_slice.T == 0, tumor_slice.T),
    cmap=colors,
    vmin=0,
    vmax=3,
    alpha=0.5,
    origin="lower",
    interpolation="nearest"
)

plt.title("Brain MRI with Tumor Segmentation")
plt.axis("off")
plt.show()
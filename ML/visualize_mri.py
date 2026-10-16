
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Patient ID
patient_id = "BraTS-GLI-00000-000"

# Path to the MRI files
data_dir = Path(__file__).resolve().parent / "data" / patient_id

# Load all four MRI modalities
modalities = ["t1n", "t1c", "t2w", "t2f"]
mri_scans = {}

for modality in modalities:
    file_path = data_dir / f"{patient_id}-{modality}.nii.gz"

    scan = nib.load(str(file_path))
    mri_scans[modality] = scan.get_fdata()

    print(f"{modality}: {mri_scans[modality].shape}")

# Load the tumor segmentation mask
seg_path = data_dir / f"{patient_id}-seg.nii.gz"
seg = nib.load(str(seg_path))
seg_data = seg.get_fdata()

# Display information
print("Segmentation dimensions:", seg_data.shape)
print("Unique tumor labels:", np.unique(seg_data))
print("MRI loaded successfully!")

# Select the MRI slice
slice_index = 65

# Create one window with six MRI images
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Display the four MRI modalities
for ax, modality in zip(axes.flat[:4], modalities):
    image = mri_scans[modality][:, :, slice_index]

    ax.imshow(image.T, cmap="gray", origin="lower")
    ax.set_title(modality.upper())
    ax.axis("off")

# Display the original FLAIR MRI
flair_slice = mri_scans["t2f"][:, :, slice_index]

axes[1, 1].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)
axes[1, 1].set_title("Original FLAIR")
axes[1, 1].axis("off")

# Extract the tumor segmentation slice
tumor_slice = seg_data[:, :, slice_index]

# Assign colors to tumor labels
# 0 = Background (hidden)
# 1 = Tumor core (red)
# 2 = Surrounding abnormal tissue (yellow)
# 3 = Enhancing tumor (blue)
colors = ListedColormap([
    "black",
    "red",
    "yellow",
    "blue"
])

# Display MRI with tumor segmentation
axes[1, 2].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1, 2].imshow(
    np.ma.masked_where(
        tumor_slice.T == 0,
        tumor_slice.T
    ),
    cmap=colors,
    vmin=0,
    vmax=3,
    alpha=0.5,
    origin="lower",
    interpolation="nearest"
)

axes[1, 2].set_title("Tumor Segmentation")
axes[1, 2].axis("off")

# Add dashboard title
plt.suptitle(
    f"Brain MRI Analysis - {patient_id} - Slice {slice_index}",
    fontsize=16
)

# Adjust layout and display everything
plt.tight_layout()
plt.show()

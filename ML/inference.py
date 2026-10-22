import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
)
from pathlib import Path

import torch
from monai.networks.nets import UNet

# Locate the trained model checkpoint
model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "brain_tumor_unet_v1.pth"
)

# Select inference device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load checkpoint onto the selected device
checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

# Recreate the U-Net architecture
model = UNet(
    **checkpoint["model_config"]
).to(device)

# Load the trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Modalities:", checkpoint["modality_order"])
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)

# Create the same preprocessing pipeline used in Colab
preprocessing = Compose([
    LoadImaged(keys=["image"]),

    EnsureChannelFirstd(keys=["image"]),

    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True
    ),

    EnsureTyped(keys=["image"])
])

# Select a local MRI patient
patient_id = "BraTS-GLI-00000-000"

data_dir = (
    Path(__file__).resolve().parent
    / "data"
    / patient_id
)

# Locate all four MRI modalities
modalities = ["t1n", "t1c", "t2w", "t2f"]

image_paths = [
    str(data_dir / f"{patient_id}-{modality}.nii.gz")
    for modality in modalities
]

# Apply preprocessing
sample = preprocessing({
    "image": image_paths
})

print("Preprocessed MRI shape:", sample["image"].shape)
print("Data type:", sample["image"].dtype)

# Add batch dimension
input_tensor = sample["image"].unsqueeze(0).to(device)

print("Running full-brain inference...")

# Switch model to evaluation mode
model.eval()

# Run inference without calculating gradients
with torch.inference_mode():
    logits = sliding_window_inference(
        inputs=input_tensor,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
        mode="gaussian"
    )

# Select the most likely tissue class for each voxel
predicted_mask = torch.argmax(logits, dim=1)

# Move prediction to CPU
predicted_mask = predicted_mask.cpu()

print("Prediction completed!")
print("Prediction shape:", predicted_mask.shape)
print("Predicted labels:", torch.unique(predicted_mask))


# --------------------------------------------------
# SAVE MODEL PREDICTION AS NIFTI
# --------------------------------------------------

# Convert predicted tensor to a 3D NumPy array
prediction_array = (
    predicted_mask[0]
    .numpy()
    .astype(np.uint8)
)

# Load the original MRI for spatial information
reference_path = data_dir / f"{patient_id}-t1n.nii.gz"
reference_mri = nib.load(str(reference_path))

# Create a NIfTI image with matching spatial coordinates
prediction_nifti = nib.Nifti1Image(
    prediction_array,
    affine=reference_mri.affine,
    header=reference_mri.header.copy()
)

# Create output folder
output_dir = Path(__file__).resolve().parent / "predictions"
output_dir.mkdir(parents=True, exist_ok=True)

# Save predicted segmentation
output_path = output_dir / f"{patient_id}-predicted-seg.nii.gz"

nib.save(prediction_nifti, str(output_path))

print("Prediction saved successfully!")
print("Output:", output_path)
print("Shape:", prediction_array.shape)

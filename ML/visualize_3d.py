
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv


# --------------------------------------------------
# 1. SELECT PATIENT AND LOAD MRI DATA
# --------------------------------------------------

patient_id = "BraTS-GLI-00000-000"
data_dir = Path(__file__).resolve().parent / "data" / patient_id

# Load tumor segmentation
seg_path = data_dir / f"{patient_id}-seg.nii.gz"
seg_image = nib.load(str(seg_path))
seg_data = np.asarray(seg_image.dataobj)

# Load structural T1 MRI
brain_path = data_dir / f"{patient_id}-t1n.nii.gz"
brain_image = nib.load(str(brain_path))
brain_data = np.asarray(brain_image.dataobj)


# --------------------------------------------------
# 2. VALIDATE MRI AND SEGMENTATION ALIGNMENT
# --------------------------------------------------

if brain_data.shape != seg_data.shape:
    raise ValueError("MRI and segmentation dimensions do not match")

if not np.allclose(brain_image.affine, seg_image.affine):
    raise ValueError("MRI and segmentation coordinates do not match")

# Create approximate brain mask from skull-stripped MRI
brain_mask = brain_data > 0

print("Brain dimensions:", brain_data.shape)
print("Brain voxels:", np.count_nonzero(brain_mask))
print("Segmentation dimensions:", seg_data.shape)
print("Tumor labels:", np.unique(seg_data))


# --------------------------------------------------
# 3. CREATE THE INTERACTIVE 3D VIEWER
# --------------------------------------------------

plotter = pv.Plotter(window_size=(1400, 900))
plotter.set_background("black")


# --------------------------------------------------
# 4. RECONSTRUCT THE BRAIN SURFACE
# --------------------------------------------------

brain_grid = pv.ImageData()
brain_grid.dimensions = np.array(brain_mask.shape) + 1
brain_grid.index_to_physical_matrix = brain_image.affine

brain_grid.cell_data["mask"] = (
    brain_mask.astype(np.uint8).flatten(order="F")
)

brain_surface = (
    brain_grid.threshold(0.5, scalars="mask")
    .extract_surface(algorithm="dataset_surface")
)

# Smooth brain surface for visualization
brain_surface = brain_surface.triangulate().smooth(
    n_iter=30,
    relaxation_factor=0.1
)

brain_actor = plotter.add_mesh(
    brain_surface,
    color="lightgray",
    opacity=0.15,
    smooth_shading=True,
    show_edges=False,
    label="Brain Surface"
)


# --------------------------------------------------
# 5. RECONSTRUCT EACH TUMOR REGION
# --------------------------------------------------

regions = {
    1: ("red", "Necrotic / Non-enhancing Core"),
    2: ("yellow", "Surrounding Abnormal Tissue"),
    3: ("blue", "Enhancing Tumor"),
}

tumor_actors = {}

for label, (color, name) in regions.items():

    # Select voxels belonging to this tumor region
    mask = (seg_data == label).astype(np.uint8)

    # Create 3D grid
    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape) + 1
    grid.index_to_physical_matrix = seg_image.affine
    grid.cell_data["mask"] = mask.flatten(order="F")

    # Extract the 3D surface
    surface = (
        grid.threshold(0.5, scalars="mask")
        .extract_surface(algorithm="dataset_surface")
    )

    if surface.n_points == 0:
        continue

    # Smooth tumor surface
    surface = surface.triangulate().smooth(
        n_iter=20,
        relaxation_factor=0.1
    )

    # Save actor for keyboard controls
    tumor_actors[label] = plotter.add_mesh(
        surface,
        color=color,
        opacity=0.65,
        smooth_shading=True,
        show_edges=False,
        label=name
    )

    print(f"Rendered: {name}")


# --------------------------------------------------
# 6. KEYBOARD CONTROLS
# --------------------------------------------------

# B key: Toggle brain visibility
def toggle_brain():
    brain_actor.SetVisibility(
        not brain_actor.GetVisibility()
    )
    plotter.render()


# 1, 2, 3 keys: Toggle tumor regions
def toggle_tumor(label):
    actor = tumor_actors.get(label)

    if actor is not None:
        actor.SetVisibility(
            not actor.GetVisibility()
        )
        plotter.render()


plotter.add_key_event("b", toggle_brain)

for label in regions:
    plotter.add_key_event(
        str(label),
        lambda selected=label: toggle_tumor(selected)
    )


# --------------------------------------------------
# 7. BUILT-IN HELP MANUAL
# --------------------------------------------------

help_text = """
3D BRAIN MRI VIEWER - CONTROLS

B    Show / Hide Brain
1    Show / Hide Tumor Core (Red)
2    Show / Hide Abnormal Tissue (Yellow)
3    Show / Hide Enhancing Tumor (Blue)

MOUSE CONTROLS
Left Drag       Rotate Brain
Mouse Wheel     Zoom In / Out
Middle Drag     Pan Camera

H    Show / Hide This Manual
"""

# Create help menu in upper-left corner
help_actor = plotter.add_text(
    help_text,
    position="upper_left",
    font_size=10,
    color="white",
    name="help_manual"
)

# H key: Toggle help menu
def toggle_help():
    help_actor.SetVisibility(
        not help_actor.GetVisibility()
    )
    plotter.render()


plotter.add_key_event("h", toggle_help)


# --------------------------------------------------
# 8. VIEWER SETTINGS
# --------------------------------------------------

# Small reminder always visible
plotter.add_text(
    "Press H for controls",
    position="lower_left",
    font_size=10,
    color="white"
)

# Display legend, axes, and coordinate grid
plotter.add_legend()
plotter.add_axes()
plotter.show_grid()

# Fit camera around entire brain
plotter.reset_camera()

# Launch interactive window
plotter.show()

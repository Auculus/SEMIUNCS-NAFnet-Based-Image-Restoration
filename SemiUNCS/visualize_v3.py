import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from SemiUNCS.Model.nafnet_v3 import NAFNetV3


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = (
    "C:/Users/prana/PycharmProjects/SEMICON/SemiUNCS/"
    "checkpoints/v3/best_model.pth"
)

DEGRADED_DIR = (
    "C:/Users/prana/PycharmProjects/SEMICON/"
    "Data/train/train/NoisyLR"
)

GROUND_TRUTH_DIR = (
    "C:/Users/prana/PycharmProjects/SEMICON/"
    "Data/train/train/GT"
)

OUTPUT_DIR = "visualizations/v3"

NUM_IMAGES = 8

SEED = 42

SCALE = 2


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target,
    max_value=1.0,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    mse = np.mean(
        (prediction - target) ** 2
    )

    if mse == 0:

        return float("inf")

    return 10.0 * np.log10(
        (max_value ** 2) / mse
    )


# ============================================================
# Load NPY
# ============================================================

def load_npy(path):

    image = np.load(
        path
    ).astype(
        np.float32
    )

    image = np.squeeze(
        image
    )

    if image.ndim != 2:

        raise ValueError(
            f"Expected 2D image, "
            f"got {image.shape}: {path}"
        )

    return image


# ============================================================
# Load V3 Model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    print(
        "Loading checkpoint..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # PyTorch 2.6+ defaults to weights_only=True.
    #
    # Your checkpoint contains additional Python objects,
    # therefore explicitly use weights_only=False.
    #
    # Only do this for a checkpoint you trust.
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Recreate EXACT V3 architecture
    # --------------------------------------------------------

    model = NAFNetV3(

        img_channel=1,

        width=32,

        enc_blk_nums=(
            2,
            2,
            4,
        ),

        middle_blk_num=4,

        dec_blk_nums=(
            2,
            2,
            2,
        ),

        scale=2,
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ) and "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove possible "module." prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Checkpoint information
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "best_psnr" in checkpoint:

            print(
                f"Best PSNR recorded: "
                f"{checkpoint['best_psnr']:.4f} dB"
            )

        elif "best_val_psnr" in checkpoint:

            print(
                f"Best validation PSNR recorded: "
                f"{checkpoint['best_val_psnr']:.4f} dB"
            )

    return model


# ============================================================
# Inference
# ============================================================

def run_inference(
    model,
    degraded,
    device,
):

    # --------------------------------------------------------
    # H,W
    #
    # ->
    #
    # 1,1,H,W
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        degraded
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            tensor
        )

    # --------------------------------------------------------
    # 1,1,H,W -> H,W
    # --------------------------------------------------------

    prediction = (
        prediction
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return prediction


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "V3 VISUALIZATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    degraded_dir = Path(
        DEGRADED_DIR
    )

    ground_truth_dir = Path(
        GROUND_TRUTH_DIR
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not degraded_dir.exists():

        raise RuntimeError(
            f"Degraded image directory "
            f"does not exist:\n"
            f"{degraded_dir}"
        )

    if not ground_truth_dir.exists():

        raise RuntimeError(
            f"Ground truth directory "
            f"does not exist:\n"
            f"{ground_truth_dir}"
        )

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    degraded_files = sorted(
        degraded_dir.glob(
            "*.npy"
        )
    )

    if len(degraded_files) == 0:

        raise RuntimeError(
            "No degraded .npy images found."
        )

    print(
        f"Available images: "
        f"{len(degraded_files)}"
    )

    # --------------------------------------------------------
    # Random selection
    # --------------------------------------------------------

    random.seed(
        SEED
    )

    selected_files = random.sample(
        degraded_files,
        min(
            NUM_IMAGES,
            len(degraded_files),
        ),
    )

    print(
        f"Visualizing "
        f"{len(selected_files)} images."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print()

    # ========================================================
    # Process images
    # ========================================================

    psnr_values = []

    for index, degraded_path in enumerate(
        selected_files,
        start=1,
    ):

        filename = (
            degraded_path.name
        )

        # ----------------------------------------------------
        # Ground truth path
        # ----------------------------------------------------

        gt_path = (
            ground_truth_dir
            / filename
        )

        if not gt_path.exists():

            print(
                f"[SKIP] Missing GT: "
                f"{filename}"
            )

            continue

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        degraded = load_npy(
            degraded_path
        )

        ground_truth = load_npy(
            gt_path
        )

        # ----------------------------------------------------
        # Run V3
        # ----------------------------------------------------

        prediction = run_inference(
            model,
            degraded,
            device,
        )

        # ----------------------------------------------------
        # Expected V3 output size
        # ----------------------------------------------------

        expected_shape = (
            degraded.shape[0] * SCALE,
            degraded.shape[1] * SCALE,
        )

        # ----------------------------------------------------
        # Verify prediction shape
        # ----------------------------------------------------

        if prediction.shape != expected_shape:

            raise RuntimeError(
                f"Unexpected model output "
                f"shape for {filename}\n"
                f"Expected: {expected_shape}\n"
                f"Got: {prediction.shape}"
            )

        # ----------------------------------------------------
        # Verify GT shape
        # ----------------------------------------------------

        if ground_truth.shape != expected_shape:

            raise RuntimeError(
                f"Unexpected GT shape "
                f"for {filename}\n"
                f"Expected: {expected_shape}\n"
                f"Got: {ground_truth.shape}"
            )

        # ----------------------------------------------------
        # Clip prediction
        # ----------------------------------------------------

        prediction_clipped = np.clip(
            prediction,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Upscale LR using nearest neighbour
        #
        # This is only for visualization.
        # It gives the LR image the same spatial size
        # as the V3 prediction and GT.
        # ----------------------------------------------------

        degraded_upscaled = np.repeat(
            np.repeat(
                degraded,
                SCALE,
                axis=0,
            ),
            SCALE,
            axis=1,
        )

        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        psnr = calculate_psnr(
            prediction_clipped,
            ground_truth,
        )

        psnr_values.append(
            psnr
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"[{index:02d}/"
            f"{len(selected_files):02d}] "
            f"{filename} | "
            f"PSNR: {psnr:.4f} dB"
        )

        # ====================================================
        # Create visualization
        # ====================================================

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        # ----------------------------------------------------
        # LR input
        # ----------------------------------------------------

        axes[0].imshow(
            degraded_upscaled,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[0].set_title(
            "Noisy LR\n"
            "(2× nearest-neighbor)"
        )

        axes[0].axis(
            "off"
        )

        # ----------------------------------------------------
        # V3 prediction
        # ----------------------------------------------------

        axes[1].imshow(
            prediction_clipped,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[1].set_title(
            f"V3 Restoration\n"
            f"PSNR: {psnr:.2f} dB"
        )

        axes[1].axis(
            "off"
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        axes[2].imshow(
            ground_truth,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[2].set_title(
            "Ground Truth"
        )

        axes[2].axis(
            "off"
        )

        # ----------------------------------------------------
        # Figure title
        # ----------------------------------------------------

        fig.suptitle(
            f"V3 Restoration — {filename}",
            fontsize=14,
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            output_dir
            / filename.replace(
                ".npy",
                ".png",
            )
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # ========================================================
    # Summary
    # ========================================================

    print()

    print("=" * 70)

    print(
        "VISUALIZATION COMPLETE"
    )

    print("=" * 70)

    if psnr_values:

        print(
            f"Images evaluated: "
            f"{len(psnr_values)}"
        )

        print(
            f"Mean PSNR: "
            f"{np.mean(psnr_values):.4f} dB"
        )

        print(
            f"Best PSNR: "
            f"{np.max(psnr_values):.4f} dB"
        )

        print(
            f"Worst PSNR: "
            f"{np.min(psnr_values):.4f} dB"
        )

    print(
        "\nResults saved to:"
    )

    print(
        output_dir.resolve()
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
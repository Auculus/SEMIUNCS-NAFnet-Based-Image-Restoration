from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):

    def __init__(
        self,
        degraded_dir,
        ground_truth_dir,
        scale=2,
        crop_size=128,
        training=True,
        filenames=None,
    ):

        self.degraded_dir = Path(degraded_dir)
        self.ground_truth_dir = Path(ground_truth_dir)

        self.scale = scale
        self.crop_size = crop_size
        self.training = training

        self.degraded_files = sorted(
            self.degraded_dir.glob("*.npy")
        )

        self.ground_truth_files = sorted(
            self.ground_truth_dir.glob("*.npy")
        )

        if len(self.degraded_files) == 0:
            raise RuntimeError(
                f"No .npy files found in:\n"
                f"{self.degraded_dir}"
            )

        if len(self.ground_truth_files) == 0:
            raise RuntimeError(
                f"No .npy files found in:\n"
                f"{self.ground_truth_dir}"
            )

        print(
            f"Found {len(self.degraded_files)} "
            f"degraded images"
        )

        print(
            f"Found {len(self.ground_truth_files)} "
            f"ground-truth images"
        )

        # Make sure the filenames match
        degraded_names = {
            f.name for f in self.degraded_files
        }

        gt_names = {
            f.name for f in self.ground_truth_files
        }

        if degraded_names != gt_names:

            missing_gt = degraded_names - gt_names
            missing_degraded = gt_names - degraded_names

            raise RuntimeError(
                "Dataset pairing mismatch!\n\n"
                f"Missing GT files: "
                f"{list(missing_gt)[:10]}\n"
                f"Missing degraded files: "
                f"{list(missing_degraded)[:10]}"
            )

        if filenames is None:

            self.filenames = sorted(
                degraded_names
            )

        else:

            self.filenames = list(filenames)

            missing = [
                f
                for f in self.filenames
                if f not in degraded_names
            ]

            if missing:
                raise RuntimeError(
                    "Some requested filenames "
                    "do not exist in the dataset:\n"
                    f"{missing[:10]}"
                )

    def __len__(self):

        return len(self.filenames)

    def load_npy(self, path):

        image = np.load(path)

        # Convert to float32
        image = image.astype(
            np.float32
        )

        # Remove unnecessary dimensions.
        #
        # Examples:
        # (1, H, W) -> (H, W)
        # (H, W, 1) -> (H, W)

        image = np.squeeze(image)

        if image.ndim != 2:

            raise ValueError(
                f"Expected a 2D grayscale image, "
                f"but got shape {image.shape} "
                f"from {path}"
            )

        return image

    def normalize(self, image):

        # IMPORTANT:
        # For now we do NOT normalize.
        #
        # We first need to inspect the actual
        # numerical range of the KLA dataset.

        return image

    def random_crop(
        self,
        degraded,
        ground_truth,
    ):

        lr_h, lr_w = degraded.shape

        crop = self.crop_size

        hr_crop = crop * self.scale

        if lr_h < crop or lr_w < crop:

            raise ValueError(
                f"Crop size {crop} is too large "
                f"for degraded image "
                f"{degraded.shape}"
            )

        top = np.random.randint(
            0,
            lr_h - crop + 1
        )

        left = np.random.randint(
            0,
            lr_w - crop + 1
        )

        degraded = degraded[
            top:top + crop,
            left:left + crop
        ]

        gt_top = top * self.scale
        gt_left = left * self.scale

        ground_truth = ground_truth[
            gt_top:gt_top + hr_crop,
            gt_left:gt_left + hr_crop
        ]

        return degraded, ground_truth

    def augment(
        self,
        degraded,
        ground_truth,
    ):

        # Horizontal flip
        if np.random.random() < 0.5:

            degraded = np.fliplr(
                degraded
            ).copy()

            ground_truth = np.fliplr(
                ground_truth
            ).copy()

        # Vertical flip
        if np.random.random() < 0.5:

            degraded = np.flipud(
                degraded
            ).copy()

            ground_truth = np.flipud(
                ground_truth
            ).copy()

        # Rotation
        k = np.random.randint(0, 4)

        if k != 0:

            degraded = np.rot90(
                degraded,
                k
            ).copy()

            ground_truth = np.rot90(
                ground_truth,
                k
            ).copy()

        return degraded, ground_truth

    def __getitem__(self, index):

        filename = self.filenames[index]

        degraded_path = (
            self.degraded_dir / filename
        )

        ground_truth_path = (
            self.ground_truth_dir / filename
        )

        # Load
        degraded = self.load_npy(
            degraded_path
        )

        ground_truth = self.load_npy(
            ground_truth_path
        )


        # Verify resolution
        expected_height = (
            degraded.shape[0] * self.scale
        )

        expected_width = (
            degraded.shape[1] * self.scale
        )

        if ground_truth.shape != (
            expected_height,
            expected_width,
        ):

            raise ValueError(
                "\nResolution mismatch!\n"
                f"File: {filename}\n"
                f"Degraded: {degraded.shape}\n"
                f"GT: {ground_truth.shape}\n"
                f"Expected GT: "
                f"{(expected_height, expected_width)}"
            )


        # Normalize
        degraded = self.normalize(
            degraded
        )

        ground_truth = self.normalize(
            ground_truth
        )

        # -----------------------------------------
        # Training augmentation
        # -----------------------------------------

        if self.training:

            degraded, ground_truth = (
                self.random_crop(
                    degraded,
                    ground_truth,
                )
            )

            degraded, ground_truth = (
                self.augment(
                    degraded,
                    ground_truth,
                )
            )

        degraded = torch.from_numpy(
            degraded.copy()
        ).float()

        ground_truth = torch.from_numpy(
            ground_truth.copy()
        ).float()

        # H,W -> C,H,W
        degraded = degraded.unsqueeze(0)

        ground_truth = ground_truth.unsqueeze(0)

        return {
            "degraded": degraded,
            "ground_truth": ground_truth,
            "filename": filename,
        }
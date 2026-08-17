import os
import random
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from SemiUNCS.Model.nafnet_v3 import build_nafnet_v3


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "Configs/v3.yaml"


# ============================================================
# GPU INFORMATION
# ============================================================

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class NPYDataset(Dataset):

    def __init__(
        self,
        degraded_dir,
        gt_dir,
        crop_size=64,
        scale=2
    ):

        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir

        self.crop_size = crop_size
        self.scale = scale

        degraded_files = {

            f
            for f in os.listdir(degraded_dir)
            if f.endswith(".npy")

        }

        gt_files = {

            f
            for f in os.listdir(gt_dir)
            if f.endswith(".npy")

        }

        self.files = sorted(
            degraded_files.intersection(
                gt_files
            )
        )

        if len(self.files) == 0:

            raise RuntimeError(
                "No matching .npy files found."
            )

        print(
            f"Dataset loaded: "
            f"{len(self.files)} image pairs"
        )

    def __len__(self):

        return len(self.files)

    # ========================================================
    # LOAD NPY
    # ========================================================

    def _load_npy(self, path):

        arr = np.load(
            path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # H x W
        # ----------------------------------------------------

        if arr.ndim == 2:

            arr = arr[None, :, :]

        # ----------------------------------------------------
        # H x W x C
        # ----------------------------------------------------

        elif arr.ndim == 3:

            if arr.shape[-1] == 1:

                arr = arr.transpose(
                    2,
                    0,
                    1
                )

            elif arr.shape[0] != 1:

                raise ValueError(
                    f"Unexpected array shape "
                    f"{arr.shape} in {path}"
                )

        else:

            raise ValueError(
                f"Expected 2D or 3D array, "
                f"got {arr.shape} in {path}"
            )

        return arr

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        filename = self.files[idx]

        degraded_path = os.path.join(
            self.degraded_dir,
            filename
        )

        gt_path = os.path.join(
            self.gt_dir,
            filename
        )

        degraded = self._load_npy(
            degraded_path
        )

        gt = self._load_npy(
            gt_path
        )

        # ----------------------------------------------------
        # Replace NaN / Inf
        # ----------------------------------------------------

        degraded = np.nan_to_num(
            degraded,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        gt = np.nan_to_num(
            gt,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        _, lr_h, lr_w = degraded.shape
        _, gt_h, gt_w = gt.shape

        expected_gt_h = lr_h * self.scale
        expected_gt_w = lr_w * self.scale

        if (
            gt_h != expected_gt_h
            or
            gt_w != expected_gt_w
        ):

            raise ValueError(
                f"\nDimension mismatch for {filename}\n"
                f"LR : {degraded.shape}\n"
                f"GT : {gt.shape}\n"
                f"Expected GT: "
                f"(1, {expected_gt_h}, {expected_gt_w})"
            )

        # ----------------------------------------------------
        # Random crop
        # ----------------------------------------------------

        crop = self.crop_size
        scale = self.scale

        if (
            lr_h >= crop
            and
            lr_w >= crop
        ):

            top = random.randint(
                0,
                lr_h - crop
            )

            left = random.randint(
                0,
                lr_w - crop
            )

            # LR crop

            degraded = degraded[
                :,
                top:top + crop,
                left:left + crop
            ]

            # Corresponding GT crop

            gt_top = top * scale
            gt_left = left * scale

            gt_crop = crop * scale

            gt = gt[
                :,
                gt_top:gt_top + gt_crop,
                gt_left:gt_left + gt_crop
            ]

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if random.random() < 0.5:

            degraded = torch.flip(
                torch.from_numpy(degraded),
                dims=[2]
            ).numpy()

            gt = torch.flip(
                torch.from_numpy(gt),
                dims=[2]
            ).numpy()

        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if random.random() < 0.5:

            degraded = torch.flip(
                torch.from_numpy(degraded),
                dims=[1]
            ).numpy()

            gt = torch.flip(
                torch.from_numpy(gt),
                dims=[1]
            ).numpy()

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        degraded = torch.from_numpy(
            degraded
        ).float()

        gt = torch.from_numpy(
            gt
        ).float()

        # ----------------------------------------------------
        # Final dimension check
        # ----------------------------------------------------

        expected_gt_size = (

            degraded.shape[1] * scale,

            degraded.shape[2] * scale

        )

        actual_gt_size = (

            gt.shape[1],

            gt.shape[2]

        )

        if actual_gt_size != expected_gt_size:

            raise RuntimeError(

                f"\nCrop dimension mismatch "
                f"for {filename}\n"

                f"LR : {degraded.shape}\n"

                f"GT : {gt.shape}\n"

                f"Expected GT spatial size: "
                f"{expected_gt_size}"

            )

        return (
            degraded,
            gt,
            filename
        )


# ============================================================
# CHARBONNIER LOSS
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(
        self,
        eps=1e-6
    ):

        super().__init__()

        self.eps = eps

    def forward(
        self,
        pred,
        target
    ):

        diff = pred - target

        loss = torch.sqrt(
            diff * diff
            +
            self.eps * self.eps
        )

        return loss.mean()


# ============================================================
# SSIM LOSS
# ============================================================

def ssim_loss(
    pred,
    target
):

    mu_x = F.avg_pool2d(
        pred,
        kernel_size=7,
        stride=1,
        padding=3
    )

    mu_y = F.avg_pool2d(
        target,
        kernel_size=7,
        stride=1,
        padding=3
    )

    sigma_x = (

        F.avg_pool2d(
            pred * pred,
            kernel_size=7,
            stride=1,
            padding=3
        )

        -

        mu_x * mu_x
    )

    sigma_y = (

        F.avg_pool2d(
            target * target,
            kernel_size=7,
            stride=1,
            padding=3
        )

        -

        mu_y * mu_y
    )

    sigma_xy = (

        F.avg_pool2d(
            pred * target,
            kernel_size=7,
            stride=1,
            padding=3
        )

        -

        mu_x * mu_y
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (

        (2 * mu_x * mu_y + C1)

        *

        (2 * sigma_xy + C2)

    )

    denominator = (

        (mu_x * mu_x + mu_y * mu_y + C1)

        *

        (sigma_x + sigma_y + C2)

    )

    ssim = numerator / (
        denominator + 1e-8
    )

    return 1.0 - ssim.mean()


# ============================================================
# GRADIENT / EDGE LOSS
# ============================================================

def gradient_loss(
    pred,
    target
):

    pred_dx = (

        pred[:, :, :, 1:]

        -

        pred[:, :, :, :-1]

    )

    target_dx = (

        target[:, :, :, 1:]

        -

        target[:, :, :, :-1]

    )

    pred_dy = (

        pred[:, :, 1:, :]

        -

        pred[:, :, :-1, :]

    )

    target_dy = (

        target[:, :, 1:, :]

        -

        target[:, :, :-1, :]

    )

    loss_x = F.l1_loss(
        pred_dx,
        target_dx
    )

    loss_y = F.l1_loss(
        pred_dy,
        target_dy
    )

    return loss_x + loss_y


# ============================================================
# V3 LOSS
# ============================================================

class V3Loss(nn.Module):

    def __init__(
        self,
        charbonnier_weight=1.0,
        ssim_weight=0.05,
        edge_weight=0.15,
        charbonnier_eps=1e-6
    ):

        super().__init__()

        self.charbonnier_weight = (
            charbonnier_weight
        )

        self.ssim_weight = (
            ssim_weight
        )

        self.edge_weight = (
            edge_weight
        )

        self.charbonnier = (
            CharbonnierLoss(
                charbonnier_eps
            )
        )

    def forward(
        self,
        pred,
        target
    ):

        loss_char = self.charbonnier(
            pred,
            target
        )

        loss_ssim = ssim_loss(
            pred,
            target
        )

        loss_edge = gradient_loss(
            pred,
            target
        )

        total = (

            self.charbonnier_weight
            *
            loss_char

            +

            self.ssim_weight
            *
            loss_ssim

            +

            self.edge_weight
            *
            loss_edge

        )

        return (
            total,
            loss_char,
            loss_ssim,
            loss_edge
        )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0

    count = 0

    for degraded, gt, _ in loader:

        degraded = degraded.to(
            device,
            non_blocking=True
        )

        gt = gt.to(
            device,
            non_blocking=True
        )

        with autocast(
            enabled=device.type == "cuda"
        ):

            pred = model(
                degraded
            )

        if pred.shape != gt.shape:

            raise RuntimeError(

                "\nValidation shape mismatch!\n"

                f"Input      : "
                f"{degraded.shape}\n"

                f"Prediction : "
                f"{pred.shape}\n"

                f"GT         : "
                f"{gt.shape}"

            )

        pred = torch.clamp(
            pred,
            0.0,
            1.0
        )

        batch_size = degraded.size(0)

        for i in range(batch_size):

            p = (

                pred[i, 0]
                .detach()
                .cpu()
                .numpy()

            )

            g = (

                gt[i, 0]
                .detach()
                .cpu()
                .numpy()

            )

            psnr = peak_signal_noise_ratio(
                g,
                p,
                data_range=1.0
            )

            ssim = structural_similarity(
                g,
                p,
                data_range=1.0
            )

            total_psnr += psnr
            total_ssim += ssim

            count += 1

    return (
        total_psnr / count,
        total_ssim / count
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    if not os.path.exists(
        checkpoint_path
    ):

        print(
            "\nWARNING:"
        )

        print(
            f"Checkpoint not found:"
            f"\n{checkpoint_path}"
        )

        print(
            "Training V3 from scratch."
        )

        return False

    print(
        "\nLoading pretrained checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        print(
            "\nERROR: Checkpoint architecture "
            "does not match NAFNetV3."
        )

        print(
            "\nThis usually means you are trying "
            "to load a V1/V2 checkpoint into V3."
        )

        print(
            "\nDetailed error:"
        )

        print(e)

        raise

    print(
        "Pretrained checkpoint loaded successfully."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    seed = config[
        "training"
    ][
        "seed"
    ]

    set_seed(
        seed
    )

    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print("=" * 70)

    print(
        "NAFNET V3 TRAINING"
    )

    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # DATA CONFIG
    # ========================================================

    scale = config[
        "data"
    ][
        "scale"
    ]

    crop_size = config[
        "data"
    ][
        "crop_size"
    ]

    print(
        f"Scale: {scale}x"
    )

    print(
        f"LR crop: "
        f"{crop_size}x{crop_size}"
    )

    print(
        f"GT crop: "
        f"{crop_size * scale}x"
        f"{crop_size * scale}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = NPYDataset(

        config["data"][
            "train_degraded_dir"
        ],

        config["data"][
            "train_ground_truth_dir"
        ],

        crop_size=crop_size,

        scale=scale
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    validation_split = config[
        "data"
    ][
        "validation_split"
    ]

    val_size = int(
        len(dataset)
        *
        validation_split
    )

    train_size = (
        len(dataset)
        -
        val_size
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_dataset, val_dataset = (
        random_split(

            dataset,

            [
                train_size,
                val_size
            ],

            generator=generator

        )
    )

    print(
        f"Training images   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images : "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    batch_size = config[
        "training"
    ][
        "batch_size"
    ]

    num_workers = config[
        "data"
    ][
        "num_workers"
    ]

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nBuilding NAFNet V3..."
    )

    model = build_nafnet_v3(

        in_channels=config[
            "model"
        ][
            "in_channels"
        ],

        out_channels=config[
            "model"
        ][
            "out_channels"
        ],

        width=config[
            "model"
        ][
            "width"
        ],

        enc_blk_nums=tuple(
            config[
                "model"
            ][
                "enc_blk_nums"
            ]
        ),

        middle_blk_num=config[
            "model"
        ][
            "middle_blk_num"
        ],

        dec_blk_nums=tuple(
            config[
                "model"
            ][
                "dec_blk_nums"
            ]
        ),

        scale=scale
    )

    model = model.to(
        device
    )

    # ========================================================
    # MODEL PARAMETER COUNT
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ========================================================
    # MODEL SHAPE TEST
    # ========================================================

    print(
        "\nChecking model dimensions..."
    )

    test_input = torch.zeros(

        1,

        config["model"][
            "in_channels"
        ],

        crop_size,

        crop_size

    ).to(device)

    with torch.no_grad():

        test_output = model(
            test_input
        )

    print(
        f"Model input : "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Model output: "
        f"{tuple(test_output.shape)}"
    )

    expected_output = (

        1,

        config["model"][
            "out_channels"
        ],

        crop_size * scale,

        crop_size * scale

    )

    if tuple(test_output.shape) != expected_output:

        raise RuntimeError(

            "\nModel output shape is incorrect!\n"

            f"Expected: {expected_output}\n"

            f"Got     : "
            f"{tuple(test_output.shape)}"

        )

    print(
        "Model dimension check: PASSED"
    )

    del test_input
    del test_output

    # ========================================================
    # PRETRAINED CHECKPOINT
    # ========================================================

    pretrained = config[
        "training"
    ].get(
        "pretrained_checkpoint",
        None
    )

    if pretrained:

        load_checkpoint(

            model,

            pretrained,

            device

        )

    else:

        print(
            "\nNo pretrained checkpoint specified."
        )

        print(
            "Training V3 from scratch."
        )

    # ========================================================
    # LOSS
    # ========================================================

    loss_fn = V3Loss(

        charbonnier_weight=config[
            "loss"
        ][
            "charbonnier_weight"
        ],

        ssim_weight=config[
            "loss"
        ][
            "ssim_weight"
        ],

        edge_weight=config[
            "loss"
        ][
            "edge_weight"
        ],

        charbonnier_eps=config[
            "loss"
        ][
            "charbonnier_eps"
        ]

    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config[
            "training"
        ][
            "learning_rate"
        ],

        weight_decay=config[
            "training"
        ][
            "weight_decay"
        ]

    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=3,

            min_lr=1e-7

        )
    )

    # ========================================================
    # AMP
    # ========================================================

    use_amp = (

        config[
            "training"
        ][
            "amp"
        ]

        and

        device.type == "cuda"

    )

    scaler = GradScaler(
        enabled=use_amp
    )

    print(
        f"AMP enabled: {use_amp}"
    )

    # ========================================================
    # SAVE DIRECTORY
    # ========================================================

    save_dir = config[
        "training"
    ][
        "save_dir"
    ]

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ========================================================
    # TRAINING SETTINGS
    # ========================================================

    epochs = config[
        "training"
    ][
        "epochs"
    ]

    best_psnr = -float("inf")

    patience = config[
        "training"
    ].get(
        "patience",
        8
    )

    epochs_without_improvement = 0

    # ========================================================
    # START TRAINING
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "STARTING NAFNET V3 TRAINING"
    )

    print("=" * 70)

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        running_loss = 0.0

        running_char = 0.0

        running_ssim = 0.0

        running_edge = 0.0

        for batch_idx, (
            degraded,
            gt,
            _
        ) in enumerate(
            train_loader
        ):

            degraded = degraded.to(

                device,

                non_blocking=True

            )

            gt = gt.to(

                device,

                non_blocking=True

            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # =================================================
            # FORWARD
            # =================================================

            with autocast(
                enabled=use_amp
            ):

                pred = model(
                    degraded
                )

                # ---------------------------------------------
                # Shape check
                # ---------------------------------------------

                if pred.shape != gt.shape:

                    raise RuntimeError(

                        "\nTraining shape mismatch!\n"

                        f"Input      : "
                        f"{degraded.shape}\n"

                        f"Prediction : "
                        f"{pred.shape}\n"

                        f"GT         : "
                        f"{gt.shape}"

                    )

                loss, loss_char, loss_ssim, loss_edge = (
                    loss_fn(
                        pred,
                        gt
                    )
                )

            # =================================================
            # BACKPROPAGATION
            # =================================================

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0

            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # =================================================
            # ACCUMULATE
            # =================================================

            running_loss += (
                loss.item()
            )

            running_char += (
                loss_char.item()
            )

            running_ssim += (
                loss_ssim.item()
            )

            running_edge += (
                loss_edge.item()
            )

        # ====================================================
        # EPOCH AVERAGES
        # ====================================================

        n = len(
            train_loader
        )

        avg_loss = (
            running_loss / n
        )

        avg_char = (
            running_char / n
        )

        avg_ssim = (
            running_ssim / n
        )

        avg_edge = (
            running_edge / n
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_psnr, val_ssim = validate(

            model,

            val_loader,

            device

        )

        scheduler.step(
            val_psnr
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"\nEpoch "
            f"{epoch:02d}/{epochs}"
        )

        print(
            f"Train Loss : "
            f"{avg_loss:.6f}"
        )

        print(
            f"  Charbonnier : "
            f"{avg_char:.6f}"
        )

        print(
            f"  SSIM        : "
            f"{avg_ssim:.6f}"
        )

        print(
            f"  Edge        : "
            f"{avg_edge:.6f}"
        )

        print(
            f"Val PSNR   : "
            f"{val_psnr:.4f} dB"
        )

        print(
            f"Val SSIM   : "
            f"{val_ssim:.6f}"
        )

        print(
            f"LR         : "
            f"{current_lr:.8f}"
        )

        # ====================================================
        # SAVE LATEST
        # ====================================================

        latest_path = os.path.join(

            save_dir,

            "latest_model.pth"

        )

        torch.save(

            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_psnr":
                    val_psnr,

                "val_ssim":
                    val_ssim

            },

            latest_path

        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            epochs_without_improvement = 0

            best_path = os.path.join(

                save_dir,

                "best_model.pth"

            )

            torch.save(

                {

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_psnr":
                        val_psnr,

                    "val_ssim":
                        val_ssim

                },

                best_path

            )

            print(

                f"*** NEW BEST MODEL: "
                f"{best_psnr:.4f} dB ***"

            )

        else:

            epochs_without_improvement += 1

            print(

                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"

            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "NAFNET V3 TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        f"Best validation PSNR: "
        f"{best_psnr:.4f} dB"
    )

    print(
        "\nBest model saved to:"
    )

    print(
        os.path.join(
            save_dir,
            "best_model.pth"
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
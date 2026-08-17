import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Charbonnier Loss
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(
        self,
        eps=1e-6,
    ):

        super().__init__()

        self.eps = eps

    def forward(
        self,
        prediction,
        target,
    ):

        diff = (
            prediction
            - target
        )

        loss = torch.sqrt(
            diff * diff
            + self.eps * self.eps
        )

        return loss.mean()


# ============================================================
# SSIM
# ============================================================

class SSIMLoss(nn.Module):

    def __init__(
        self,
        window_size=11,
        sigma=1.5,
    ):

        super().__init__()

        self.window_size = (
            window_size
        )

        self.sigma = sigma

    def gaussian_window(
        self,
        channel,
        device,
        dtype,
    ):

        coords = torch.arange(
            self.window_size,
            device=device,
            dtype=dtype,
        )

        coords = (
            coords
            - self.window_size // 2
        )

        gaussian = torch.exp(
            -(coords ** 2)
            / (2 * self.sigma ** 2)
        )

        gaussian = (
            gaussian
            / gaussian.sum()
        )

        window = (
            gaussian[:, None]
            * gaussian[None, :]
        )

        window = window.unsqueeze(
            0
        ).unsqueeze(0)

        window = window.expand(
            channel,
            1,
            self.window_size,
            self.window_size,
        )

        return window

    def forward(
        self,
        prediction,
        target,
    ):

        channel = prediction.size(
            1
        )

        window = self.gaussian_window(
            channel,
            prediction.device,
            prediction.dtype,
        )

        padding = (
            self.window_size // 2
        )

        mu_prediction = F.conv2d(
            prediction,
            window,
            padding=padding,
            groups=channel,
        )

        mu_target = F.conv2d(
            target,
            window,
            padding=padding,
            groups=channel,
        )

        mu_prediction_sq = (
            mu_prediction
            * mu_prediction
        )

        mu_target_sq = (
            mu_target
            * mu_target
        )

        mu_prediction_target = (
            mu_prediction
            * mu_target
        )

        sigma_prediction_sq = (
            F.conv2d(
                prediction * prediction,
                window,
                padding=padding,
                groups=channel,
            )
            - mu_prediction_sq
        )

        sigma_target_sq = (
            F.conv2d(
                target * target,
                window,
                padding=padding,
                groups=channel,
            )
            - mu_target_sq
        )

        sigma_prediction_target = (
            F.conv2d(
                prediction * target,
                window,
                padding=padding,
                groups=channel,
            )
            - mu_prediction_target
        )

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        numerator_1 = (
            2
            * mu_prediction_target
            + C1
        )

        numerator_2 = (
            2
            * sigma_prediction_target
            + C2
        )

        denominator_1 = (
            mu_prediction_sq
            + mu_target_sq
            + C1
        )

        denominator_2 = (
            sigma_prediction_sq
            + sigma_target_sq
            + C2
        )

        ssim_map = (
            numerator_1
            * numerator_2
            /
            (
                denominator_1
                * denominator_2
            )
        )

        ssim = ssim_map.mean()

        return 1.0 - ssim


# ============================================================
# Combined Restoration Loss
# ============================================================

class RestorationLoss(nn.Module):

    def __init__(
        self,
        charbonnier_eps=1e-6,
        charbonnier_weight=1.0,
        ssim_weight=0.1,
    ):

        super().__init__()

        self.charbonnier_weight = (
            charbonnier_weight
        )

        self.ssim_weight = (
            ssim_weight
        )

        self.charbonnier = (
            CharbonnierLoss(
                eps=charbonnier_eps
            )
        )

        self.ssim = (
            SSIMLoss()
        )

    def forward(
        self,
        prediction,
        target,
    ):

        charbonnier_loss = (
            self.charbonnier(
                prediction,
                target,
            )
        )

        ssim_loss = (
            self.ssim(
                prediction,
                target,
            )
        )

        total_loss = (
            self.charbonnier_weight
            * charbonnier_loss
            +
            self.ssim_weight
            * ssim_loss
        )

        return {

            "total":
                total_loss,

            "charbonnier":
                charbonnier_loss,

            "ssim":
                ssim_loss,
        }
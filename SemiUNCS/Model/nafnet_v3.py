import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SimpleGate
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(2, dim=1)

        return x1 * x2


# ============================================================
# Simplified Channel Attention
#
# Instead of the V2:
#
# AdaptiveAvgPool
# -> Conv
# -> GELU
# -> Conv
# -> Sigmoid
#
# we use the lightweight NAFNet-style SCA.
# ============================================================

class SimplifiedChannelAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        attention = self.pool(x)

        attention = self.conv(attention)

        return x * attention


# ============================================================
# NAF Block V3
# ============================================================

class NAFBlockV3(nn.Module):

    def __init__(
        self,
        channels,
        expansion=2,
        dropout=0.0
    ):

        super().__init__()

        hidden = channels * expansion

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        self.norm1 = nn.GroupNorm(
            1,
            channels
        )

        self.conv1 = nn.Conv2d(
            channels,
            hidden * 2,
            kernel_size=1
        )

        self.conv2 = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2
        )

        self.sg = SimpleGate()

        # ----------------------------------------------------
        # V3: simplified channel attention
        # ----------------------------------------------------

        self.sca = SimplifiedChannelAttention(
            hidden
        )

        self.conv3 = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1
        )

        self.dropout1 = nn.Dropout2d(
            dropout
        )

        # ----------------------------------------------------
        # Learnable residual scaling
        # ----------------------------------------------------

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

        # ----------------------------------------------------
        # Feed-forward branch
        # ----------------------------------------------------

        self.norm2 = nn.GroupNorm(
            1,
            channels
        )

        self.ffn = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden * 2,
                kernel_size=1
            ),

            SimpleGate(),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1
            )
        )

        self.dropout2 = nn.Dropout2d(
            dropout
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        # ====================================================
        # Spatial mixing
        # ====================================================

        residual = x

        y = self.norm1(x)

        y = self.conv1(y)

        y = self.conv2(y)

        y = self.sg(y)

        y = self.sca(y)

        y = self.conv3(y)

        y = self.dropout1(y)

        x = residual + y * self.beta

        # ====================================================
        # Feed-forward
        # ====================================================

        residual = x

        y = self.norm2(x)

        y = self.ffn(y)

        y = self.dropout2(y)

        x = residual + y * self.gamma

        return x


# ============================================================
# PixelShuffle Upsampler
# ============================================================

class PixelShuffleUpsampler(nn.Module):

    def __init__(
        self,
        channels,
        scale=2
    ):

        super().__init__()

        if scale == 2:

            self.body = nn.Sequential(

                nn.Conv2d(
                    channels,
                    channels * 4,
                    kernel_size=3,
                    padding=1
                ),

                nn.PixelShuffle(2)
            )

        else:

            raise ValueError(
                "V3 currently supports scale=2 only."
            )

    def forward(self, x):

        return self.body(x)


# ============================================================
# NAFNet V3
# ============================================================

class NAFNetV3(nn.Module):

    def __init__(
        self,
        img_channel=1,
        width=32,
        enc_blk_nums=(2, 2, 4),
        middle_blk_num=4,
        dec_blk_nums=(2, 2, 2),
        scale=2
    ):

        super().__init__()

        self.scale = scale

        # ====================================================
        # Input
        # ====================================================

        self.intro = nn.Conv2d(
            img_channel,
            width,
            kernel_size=3,
            padding=1
        )

        # ====================================================
        # Encoder
        # ====================================================

        self.encoders = nn.ModuleList()

        self.downs = nn.ModuleList()

        channels = width

        for num_blocks in enc_blk_nums:

            blocks = [

                NAFBlockV3(
                    channels
                )

                for _ in range(num_blocks)
            ]

            self.encoders.append(
                nn.Sequential(*blocks)
            )

            self.downs.append(

                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2
                )
            )

            channels *= 2

        # ====================================================
        # Middle
        # ====================================================

        self.middle = nn.Sequential(

            *[
                NAFBlockV3(
                    channels
                )

                for _ in range(
                    middle_blk_num
                )
            ]
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.ups = nn.ModuleList()

        self.decoders = nn.ModuleList()

        for num_blocks in dec_blk_nums:

            self.ups.append(

                nn.Sequential(

                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=1
                    ),

                    nn.PixelShuffle(2)
                )
            )

            channels //= 2

            blocks = [

                NAFBlockV3(
                    channels
                )

                for _ in range(num_blocks)
            ]

            self.decoders.append(
                nn.Sequential(*blocks)
            )

        # ====================================================
        # Feature reconstruction
        # ====================================================

        self.pre_shuffle = nn.Sequential(

            nn.Conv2d(
                width,
                width,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )

        self.pixel_shuffle = PixelShuffleUpsampler(
            width,
            scale=2
        )

        self.output = nn.Conv2d(
            width,
            img_channel,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        baseline = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x = self.intro(x)

        skips = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):

            x = encoder(x)

            skips.append(x)

            x = down(x)

        # ----------------------------------------------------
        # Middle
        # ----------------------------------------------------

        x = self.middle(x)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        for up, decoder, skip in zip(
            self.ups,
            self.decoders,
            reversed(skips)
        ):

            x = up(x)

            x = x + skip

            x = decoder(x)

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        x = self.pre_shuffle(x)

        x = self.pixel_shuffle(x)

        x = self.output(x)

        # ----------------------------------------------------
        # Residual reconstruction
        # ----------------------------------------------------

        x = x + baseline

        return x


# ============================================================
# Factory
# ============================================================

def build_nafnet_v3(
    in_channels=1,
    out_channels=1,
    width=32,
    enc_blk_nums=(2, 2, 4),
    middle_blk_num=4,
    dec_blk_nums=(2, 2, 2),
    scale=2
):

    model = NAFNetV3(

        img_channel=in_channels,

        width=width,

        enc_blk_nums=enc_blk_nums,

        middle_blk_num=middle_blk_num,

        dec_blk_nums=dec_blk_nums,

        scale=scale
    )

    return model
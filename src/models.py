import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """
    Spatial Transformer Network.

    Predicts an affine 2x3 transformation matrix and applies it to the input.
    Used here as a learned target alignment module.
    """

    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 6),
        )

        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        batch_size = x.size(0)

        xs = self.localization(x)
        xs = xs.view(batch_size, -1)

        theta = self.fc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)

        return x_transformed, theta


class HDRNet(nn.Module):
    """
    Lightweight HDRNet-style generator.

    It predicts local and global residual corrections and applies them to the input.
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 3, grid_size: int = 16):
        super().__init__()

        self.grid_size = grid_size

        self.local_branch = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

        self.global_branch = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.global_fc = nn.Linear(64, output_nc)

    def forward(self, x):
        local_grid = self.local_branch(x)

        local_adjust = F.interpolate(
            local_grid,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        batch_size = x.size(0)

        global_feat = self.global_branch(x)
        global_feat = global_feat.view(batch_size, -1)

        global_adjust = self.global_fc(global_feat)
        global_adjust = global_adjust.unsqueeze(2).unsqueeze(3)
        global_adjust = global_adjust.expand_as(local_adjust)

        adjustment = local_adjust + global_adjust

        out = x + adjustment

        return torch.tanh(out)


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    Returns intermediate feature maps and final logits.
    """

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        kw = 4
        padw = 1

        first_layer = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        self.layers = nn.ModuleList([nn.Sequential(*first_layer)])

        nf_mult = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                    ),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            )
        )

        self.output_layer = nn.Conv2d(
            ndf * nf_mult,
            1,
            kernel_size=kw,
            stride=1,
            padding=padw,
        )

    def forward(self, input_tensor):
        result = []
        x = input_tensor

        for layer in self.layers:
            x = layer(x)
            result.append(x)

        out = self.output_layer(x)
        result.append(out)

        return result


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator.
    Applies several PatchGAN discriminators at progressively downsampled scales.
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        num_D: int = 3,
    ):
        super().__init__()

        self.num_D = num_D

        self.discriminators = nn.ModuleList(
            [
                NLayerDiscriminator(input_nc, ndf, n_layers)
                for _ in range(num_D)
            ]
        )

        self.downsample = nn.AvgPool2d(
            3,
            stride=2,
            padding=[1, 1],
            count_include_pad=False,
        )

    def forward(self, input_tensor):
        results = []
        x = input_tensor

        for discriminator in self.discriminators:
            results.append(discriminator(x))
            x = self.downsample(x)

        return results
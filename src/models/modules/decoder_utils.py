# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class Dec_linear(nn.Module):
    def __init__(self, n_in, n_out):
        self.decoder = nn.Sequential(nn.Linear(n_in, n_out))

    def forward(self, x):
        return self.decoder(x)


class Dec_nonlinear(nn.Module):
    def __init__(self, n_in, n_out):
        self.decoder = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.decoder(x)


class Dec_map_deconv64(nn.Module):
    def __init__(self, n_in):
        self.map_decoder = nn.Sequential(
            nn.Linear(n_in, 4096),
            nn.ReLU(),
            Reshape(16, 16, 16),
            MapDecoder_2x_Deconv(16),
        )

    def forward(self, x):
        return self.map_decoder(x)


### AUXILIARY FUNCTIONS


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class MapDecoder_2x_Deconv(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()

        # The parameters for ConvTranspose2D are from the PyTorch repo.
        # Ref: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # Ref: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # Ref: https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/13
        # Ref: (padding) https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
        self.decoder = nn.Sequential(
            ConvTranspose2d_FixOutputSize(
                nn.ConvTranspose2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
                output_size=(32, 32),
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            ConvTranspose2d_FixOutputSize(
                nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1),
                output_size=(64, 64),
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)


class ConvTranspose2d_FixOutputSize(nn.Module):
    """A wrapper to fix the output size of ConvTranspose2D.

    Ref: https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/13
    Ref: (other alternatives) https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, conv, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x

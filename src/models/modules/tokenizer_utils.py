# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "ResNet2d3d_full",
    "resnet18_2d3d_full",
    "resnet34_2d3d_full",
    "resnet50_2d3d_full",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def select_resnet(network, norm="bn"):
    param = {"feature_size": 1024}
    # backbones = {}
    if network == "resnet18":
        model_img = resnet18_2d3d_full(track_running_stats=True, norm=norm)
        model_flow = resnet18_2d3d_full_C2(track_running_stats=True, norm=norm)
        model_seg = resnet18_2d3d_full_C1(track_running_stats=True, norm=norm)
        model_depth = resnet18_2d3d_full_C1(track_running_stats=True, norm=norm)

        param["feature_size"] = 256
    elif network == "resnet34":
        model = resnet34_2d3d_full(track_running_stats=True, norm=norm)
        param["feature_size"] = 256
    elif network == "resnet50":
        model = resnet50_2d3d_full(track_running_stats=True, norm=norm)
    else:
        raise NotImplementedError

    return model_img, model_seg, model_depth, model_flow, param
    # return backbones, param


class PCL_encoder(nn.Module):
    def __init__(self, feat_size):
        """
        data_len (int)      : 2 for X and Y, 3 to include polarity as well
        latent_size (int)            : Size of latent vector
        tc (bool)           : True if temporal coding is included
        params (list)       : Currently just the image resolution (H, W)
        """
        super().__init__()

        # input is tensor of size [B, transformer_hist_length, num_points(720 max), 2(XY)]
        # output is of size [B, transformer_hist_length, feat_size]
        self.feat_size = feat_size

        self.featnet = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, feat_size)
        )

        self.weight_init()

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weight_init(self):
        self.featnet.apply(self.kaiming_init)
        self.encoder.apply(self.kaiming_init)

    def forward(self, x, times=None):
        # ECN computes per-event spatial features
        x = self.featnet(x)

        # Symmetric function to reduce N features to 1 a la PointNet
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        # Compress to latent space
        embedding = self.encoder(x)

        return embedding


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        clip_len=1,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            clip_len, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, clip_len=1, **kwargs):
    model = ResNet(block, layers, clip_len=clip_len, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18_custom(pretrained=False, clip_len=1, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        clip_len=clip_len,
        **kwargs
    )


def resnet50_custom(pretrained=False, clip_len=1, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        clip_len=clip_len,
        **kwargs
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=bias,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class MyLayerNorm(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = super().forward(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
        norm="bn",
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        if norm == "bn":
            self.n1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(planes)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        if norm == "bn":
            self.n2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n2 = MyLayerNorm(planes)
        elif norm == "none":
            self.n2 = nn.Identity()

        self.downsample = downsample
        self.stride = stride
        self.norm = norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
        norm="bn",
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        if norm == "bn":
            self.n1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(planes)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        if norm == "bn":
            self.n2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n2 = MyLayerNorm(planes)
        elif norm == "none":
            self.n2 = nn.Identity()

        self.downsample = downsample
        self.stride = stride
        self.norm = norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
        norm="bn",
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        if norm == "bn":
            self.n1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(planes)
        elif norm == "none":
            self.n2 = nn.Identity()

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        if norm == "bn":
            self.n2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n2 = MyLayerNorm(planes)
        elif norm == "none":
            self.n2 = nn.Identity()

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        if norm == "bn":
            self.n3 = nn.BatchNorm3d(
                planes * 4, track_running_stats=track_running_stats
            )
        elif norm == "ln":
            self.n3 = MyLayerNorm(planes * 4)
        elif norm == "none":
            self.n3 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.norm = norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.n3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
        norm="bn",
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        if norm == "bn":
            self.n1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(planes)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=bias,
        )
        if norm == "bn":
            self.n2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n2 = MyLayerNorm(planes)
        elif norm == "none":
            self.n2 = nn.Identity()

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        if norm == "bn":
            self.n3 = nn.BatchNorm3d(
                planes * 4, track_running_stats=track_running_stats
            )
        elif norm == "ln":
            self.n3 = MyLayerNorm(planes * 4)
        elif norm == "none":
            self.n3 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.norm = norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.n3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class ResNet2d3d_full(nn.Module):
    def __init__(self, block, layers, track_running_stats=True, norm="bn"):
        super().__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias
        )

        if norm == "bn":
            self.n1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(64)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        if not isinstance(block, list):
            block = [block] * 4

        self.norm = norm

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block[3], 256, layers[3], stride=2, is_final=True
        )
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            if self.norm == "bn":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(
                        planes * block.expansion,
                        track_running_stats=self.track_running_stats,
                    ),
                )
            elif self.norm == "ln":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    MyLayerNorm(planes * block.expansion),
                )
            elif self.norm == "none":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                track_running_stats=self.track_running_stats,
                norm=self.norm,
            )
        )
        self.inplanes = planes * block.expansion
        if is_final:  # if is final block, no ReLU in the final output
            for i in range(1, blocks - 1):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    track_running_stats=self.track_running_stats,
                    use_final_relu=False,
                    norm=self.norm,
                )
            )
        else:
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet2d3d_full_C1(nn.Module):
    def __init__(self, block, layers, track_running_stats=True, norm="bn"):
        super().__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias
        )

        if norm == "bn":
            self.n1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(64)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        if not isinstance(block, list):
            block = [block] * 4

        self.norm = norm

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block[3], 256, layers[3], stride=2, is_final=True
        )
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            if self.norm == "bn":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(
                        planes * block.expansion,
                        track_running_stats=self.track_running_stats,
                    ),
                )
            elif self.norm == "ln":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    MyLayerNorm(planes * block.expansion),
                )
            elif self.norm == "none":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                track_running_stats=self.track_running_stats,
                norm=self.norm,
            )
        )
        self.inplanes = planes * block.expansion
        if is_final:  # if is final block, no ReLU in the final output
            for i in range(1, blocks - 1):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    track_running_stats=self.track_running_stats,
                    use_final_relu=False,
                    norm=self.norm,
                )
            )
        else:
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet2d3d_full_C2(nn.Module):
    def __init__(self, block, layers, track_running_stats=True, norm="bn"):
        super().__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(
            2, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias
        )

        if norm == "bn":
            self.n1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        elif norm == "ln":
            self.n1 = MyLayerNorm(64)
        elif norm == "none":
            self.n1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        if not isinstance(block, list):
            block = [block] * 4

        self.norm = norm

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block[3], 256, layers[3], stride=2, is_final=True
        )
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            if self.norm == "bn":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(
                        planes * block.expansion,
                        track_running_stats=self.track_running_stats,
                    ),
                )
            elif self.norm == "ln":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                    MyLayerNorm(planes * block.expansion),
                )
            elif self.norm == "none":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=customized_stride,
                        bias=False,
                    ),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                track_running_stats=self.track_running_stats,
                norm=self.norm,
            )
        )
        self.inplanes = planes * block.expansion
        if is_final:  # if is final block, no ReLU in the final output
            for i in range(1, blocks - 1):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    track_running_stats=self.track_running_stats,
                    use_final_relu=False,
                    norm=self.norm,
                )
            )
        else:
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                        norm=self.norm,
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18_2d3d_full(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet2d3d_full(
        [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [2, 2, 2, 2], **kwargs
    )
    return model


def resnet18_2d3d_full_C1(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet2d3d_full_C1(
        [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [2, 2, 2, 2], **kwargs
    )
    return model


def resnet18_2d3d_full_C2(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet2d3d_full_C2(
        [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [2, 2, 2, 2], **kwargs
    )
    return model


def resnet34_2d3d_full(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet2d3d_full(
        [BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [3, 4, 6, 3], **kwargs
    )
    return model


def resnet50_2d3d_full(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet2d3d_full(
        [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], [3, 4, 6, 3], **kwargs
    )
    return model

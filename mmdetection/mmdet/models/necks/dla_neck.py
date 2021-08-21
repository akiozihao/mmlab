import math

import numpy as np
# from CenterTrack.src.lib.model.networks.DCNv2.dcn_v2 import DCN
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS
from torch import nn
from mmcv.ops import ModulatedDeformConv2dPack


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


# class DeformConv(nn.Module):
#     def __init__(self, chi, cho):
#         super(DeformConv, self).__init__()
#         self.actf = nn.Sequential(
#             nn.BatchNorm2d(cho, momentum=0.1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv = ModulatedDeformConv2dPack(
#             in_channels=chi,
#             out_channels=cho,
#             kernel_size=(3, 3),
#             stride=1,
#             padding=1,
#             dilation=1,
#             groups=1,
#             deform_groups=1,
#             bias=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.actf(x)
#         return x


class IDAUp(BaseModule):
    def __init__(self,
                 planes,
                 channels,
                 up_f,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=None
                 ):
        super(IDAUp, self).__init__(init_cfg)
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            # todo check mmtracking dcnv2
            if use_dcn:
                proj = DeformConv(c, planes)
                node = DeformConv(planes, planes)
            else:
                proj = ConvModule(
                    c,
                    planes,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True
                )
                node = ConvModule(
                    planes,
                    planes,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=True
                )

            up = build_conv_layer(
                dict(type='deconv'),
                planes,
                planes,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=planes,
                bias=False,
            )

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()


class DLAUp(BaseModule):
    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=None):
        super(DLAUp, self).__init__(init_cfg)
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          use_dcn=use_dcn,
                          conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg,
                          init_cfg=init_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@NECKS.register_module()
class DLANeck(BaseModule):

    def __init__(self,
                 channels,
                 down_ratio,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=None):
        super(DLANeck, self).__init__(init_cfg)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level,
            channels[self.first_level:],
            scales,
            use_dcn=use_dcn,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg
        )
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
            use_dcn=use_dcn,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg
        )

    def forward(self, x):
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]

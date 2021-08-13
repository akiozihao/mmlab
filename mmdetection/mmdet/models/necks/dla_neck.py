import numpy as np
from mmcv.ops import ModulatedDeformConv2dPack
from mmdet.models.builder import NECKS
from torch import nn
import math
BN_MOMENTUM = 0.1
from CenterTrack.src.lib.model.networks.DCNv2.dcn_v2 import DCN

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

# class DeformConv(nn.Module):
#     def __init__(self, chi, cho):
#         super(DeformConv, self).__init__()
#         self.actf = nn.Sequential(
#             nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
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


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type[0](c, o)
            node = node_type[1](o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

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


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None,
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
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
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_NODE = {
    'dcn': (DeformConv, DeformConv),
}


@NECKS.register_module()
class DLANeck(nn.Module):
    arch_settings = {
        34: ([16, 32, 64, 128, 256, 512], 4)
    }

    def __init__(self, arch):
        super(DLANeck, self).__init__()
        self.node_type = DLA_NODE['dcn']
        assert arch == 34, 'Only support dla-34.'
        channels = self.arch_settings[arch][0]
        down_ratio = self.arch_settings[arch][1]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level,
            channels[self.first_level:],
            scales, node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type
        )

    def forward(self, x):
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]

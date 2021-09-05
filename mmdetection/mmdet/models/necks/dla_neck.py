import math

import numpy as np
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS


class IDAUp(BaseModule):
    """Iterative Deep Aggregation Layer

    Args:
        planes (int): Number of output channels.
        channels (List[int]): Number of input channels.
        up_f (List[int]): Upsample factor.
        use_dcn (boolt): Whether to use dcn. Default True.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 planes,
                 channels,
                 up_f,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None
                 ):

        super(IDAUp, self).__init__(init_cfg)
        self.use_dcn = use_dcn
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = ConvModule(
                c,
                planes,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if use_dcn else conv_cfg,
                norm_cfg=norm_cfg,
                bias=False
            )
            node = ConvModule(
                planes,
                planes,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if use_dcn else conv_cfg,
                norm_cfg=norm_cfg,
                bias=False
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
    """
    Args:
        startp (int): Layer number to start, which controls how many IDA layers participant in forward.
        channels (list[int]): Number of channels.
        scales (list[int]): Upsample factors.
        in_channels (int): Number of input channels. Default None.
        use_dcn (boolt): Whether to use dcn. Default True.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=None,
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
                    IDAUp(channels[j],
                          in_channels[j:],
                          scales[j:] // scales[j],
                          use_dcn=use_dcn,
                          conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg))
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
    """Deep Layer Aggregation Neck

    Args:
        channels (list[int]): Number of input channels.
        down_ratio (int): Downsample ratio.
        use_dcn (boolt): Whether to use dcn. Default True.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 down_ratio,
                 use_dcn=True,
                 conv_cfg=None,
                 norm_cfg=None,
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
            norm_cfg=norm_cfg
        )
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
            use_dcn=use_dcn,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]

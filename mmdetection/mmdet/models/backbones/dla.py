import torch
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn

from mmdet.models.builder import BACKBONES


class DLABasicBlock(BaseModule):
    """BasicBlock for Deep Layer Aggregation.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int | tuple[int]): Stride of the conv1.
        dilation (int | tuple[int]: Spacing between kernel elements.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        conv_cfg=None,
        norm_cfg=None,
        init_cfg=None,
    ):
        super(DLABasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(BaseModule):
    """Root layer of the tree structure in Deep Layer Aggregation.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        kernel_size (int | tuple[int]): Size of the convolution kernel.
        residual (bool): Whether to use residual.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 residual,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super(Root, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.conv = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.add_module(self.norm1_name, norm1)
        self.residual = residual
        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm1(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(BaseModule):
    """Tree layer of the tree structure in Deep Layer Aggregation.

    Args:
        levels (int): The level of current tree.
        block (type): Block type.
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int | tuple[int]): Stride of the convolution. Default 1.
        level_root (bool): Whether to use level_root. Default False.
        root_dim (int): Root dimension. Default 0.
        root_kernel_size (int | tuple[int]):  Size of the root convolution
            kernel. Default 1.
        dilation (int | tuple[int]: Spacing between kernel elements.
        root_residual (bool): Whether to use residual. Default False.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 levels,
                 block,
                 inplanes,
                 planes,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):

        super(Tree, self).__init__(init_cfg)
        self.norm_cfg = norm_cfg
        if root_dim == 0:
            root_dim = 2 * planes
        if level_root:
            root_dim += inplanes
        if levels == 1:
            self.tree1 = block(
                inplanes,
                planes,
                stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg)
            self.tree2 = block(
                planes,
                planes,
                1,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                inplanes,
                planes,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg)
            self.tree2 = Tree(
                levels - 1,
                block,
                planes,
                planes,
                root_dim=root_dim + planes,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg)
        if levels == 1:
            self.root = Root(
                root_dim,
                planes,
                root_kernel_size,
                root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=init_cfg)

        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if inplanes != planes:
            conv = build_conv_layer(
                conv_cfg, inplanes, planes, 1, stride=1, bias=False)
            norm_name, norm = build_norm_layer(norm_cfg, planes)
            self.project = nn.Sequential(conv, norm)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLA(BaseModule):
    """Deep Layer Aggregation.

    Args:
        levels (list[int]): Number of convolution layers at each level.
        channels (list[int]): Number of channels at each level.
        block (type): Block type. Default DLABasicBlock.
        root_residual (bool): Whether to use residual. Default False.
        use_pre_img (bool): Whether to use pre img. Default False.
        use_pre_hm (bool): Whether to use pre heatmap. Default False.
        conv_cfg (dict): Config dict for convolution layers. Default: None,
        which means using Conv2d.
        norm_cfg (dict): Config dict for normalization layers. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 levels,
                 channels,
                 block=DLABasicBlock,
                 root_residual=False,
                 use_pre_img=True,
                 use_pre_hm=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):

        super(DLA, self).__init__(init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.channels = channels
        self.base_layer = ConvModule(
            3,
            channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=root_residual,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=root_residual,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=root_residual,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=root_residual,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        if use_pre_img:
            self.pre_img_layer = ConvModule(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            )
        if use_pre_hm:
            self.pre_hm_layer = ConvModule(
                1,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(
                ConvModule(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

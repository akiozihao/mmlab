import torch
from torch import nn

from mmcv.runner import BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.builder import BACKBONES


BN_MOMENTUM = 0.1


# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
#                                stride=stride, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=dilation,
#                                bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
# We need new class because forward need an explicit residual argument
class DLABasicBlock(BasicBlock):
    def forward(self, x, residual=None):
        if self.with_cp and x.requires_grad:
            raise NotImplementedError("DLABasicBlock doesn't implement with_cp.")
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)  # todo use norm_cfg?
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, norm_cfg=dict(type='BN', momentum=BN_MOMENTUM)):
        super(Tree, self).__init__()
        self.norm_cfg = norm_cfg
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, norm_cfg=self.norm_cfg)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, norm_cfg=self.norm_cfg)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

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
class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=DLABasicBlock, residual_root=False,
                 use_pre_img=True, use_pre_hm=True, norm_cfg=dict(type='BN', momentum=BN_MOMENTUM)):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, norm_cfg=norm_cfg,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, norm_cfg=norm_cfg,
                           root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, norm_cfg=norm_cfg,
                           root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, norm_cfg=norm_cfg,
                           root_residual=residual_root)
        if use_pre_img:
            self.pre_img_layer = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                          padding=3, bias=False),
                nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
        if use_pre_hm:
            self.pre_hm_layer = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                          padding=3, bias=False),
                nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    # def _make_level(self, block, inplanes, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or inplanes != planes:
    #         downsample = nn.Sequential(
    #             nn.MaxPool2d(stride, stride=stride),
    #             nn.Conv2d(inplanes, planes,
    #                       kernel_size=1, stride=1, bias=False),
    #             nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
    #         )
    #
    #     layers = []
    #     layers.append(block(inplanes, planes, stride, downsample=downsample))
    #     for i in range(1, blocks):
    #         layers.append(block(inplanes, planes))
    #
    #     return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
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

#     def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
#         # fc = self.fc
#         if name.endswith('.pth'):
#             model_weights = torch.load(data + name)
#         else:
#             model_url = get_model_url(data, name, hash)
#             model_weights = model_zoo.load_url(model_url)
#         num_classes = len(model_weights[list(model_weights.keys())[-1]])
#         self.fc = nn.Conv2d(
#             self.channels[-1], num_classes,
#             kernel_size=1, stride=1, padding=0, bias=True)
#         self.load_state_dict(model_weights, strict=False)
#         # self.fc = fc
#
#
# def dla34(pretrained=True, **kwargs):  # DLA-34
#     model = DLA([1, 1, 1, 2, 2, 1],
#                 [16, 32, 64, 128, 256, 512],
#                 block=BasicBlock, **kwargs)
#     if pretrained:
#         model.load_pretrained_model(
#             data='imagenet', name='dla34', hash='ba72cf86')
#     else:
#         print('Warning: No ImageNet pretrain!!')
#     return model

import torch
from mmdet.models.backbones.dla import DLA
from mmdet.models.necks.dla_neck import DLANeck


neck_pth = '/home/akio/Downloads/crowdhuman_split/neck.pt'

neck = DLANeck(34).cuda()

x = torch.randn(1, 3, 544, 960).cuda()
pre_img = torch.randn(1, 3, 544, 960).cuda()
pre_hm = torch.randn(1, 1, 544, 960).cuda()
backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512]).cuda()
backbone_out = backbone(x, pre_img, pre_hm)

neck_out = neck(backbone_out)  # tensor
print('neck_out.shape',neck_out.shape)
print('done')

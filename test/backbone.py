
import torch
from mmdet.models.backbones.dla import DLA
from test_utils import Struct

from CenterTrack.src.lib.model.networks.dla import DLA as DLA_Ori
from CenterTrack.src.lib.model.networks.dla import BasicBlock

backbone_path_ori = '../mmpth/test_o_backbone.pt'
backbone_path = '../mmpth/test_mm_backbone.pt'
opt = Struct(pre_img=True, pre_hm=True)
use_cuda = True

backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256,512],
               norm_cfg=dict(type='BN', momentum=0.1))
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                       block=BasicBlock,
                       opt=opt)
st_ori = torch.load(backbone_path_ori)
st = torch.load(backbone_path)

backbone.load_state_dict(st)
backbone_ori.load_state_dict(st_ori)
assert all([(v1 == v2).all()
            for v1, v2 in zip(backbone.parameters(), backbone_ori.parameters())
            ])

x = torch.randn(1, 3, 544, 960)
pre_img = torch.randn(1, 3, 544, 960)
pre_hm = torch.randn(1, 1, 544, 960)

if use_cuda:
    backbone = backbone.cuda()
    backbone_ori = backbone_ori.cuda()
    x = x.cuda()
    pre_img = pre_img.cuda()
    pre_hm = pre_hm.cuda()

output1 = backbone(x, pre_img, pre_hm)
output2 = backbone_ori(x, pre_img, pre_hm)
assert all([(v1 == v2).all() for v1, v2 in zip(output1, output2)])

print('done')

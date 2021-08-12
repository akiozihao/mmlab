import torch
from mmdet.models.backbones.dla import DLA

from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLA as DLA_Ori

backbone_path = '../tensors/backbone.pt'
# backbone_path = '/home/akio/Downloads/new_crowdhuman.pth'


backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512])
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1],
                       [16, 32, 64, 128, 256, 512],
                       block=BasicBlock)
st = torch.load(backbone_path)

backbone.load_state_dict(st)
backbone_ori.load_state_dict(st)
assert all([(v1 == v2).all() for v1, v2 in zip(backbone.parameters(), backbone_ori.parameters())])
x = torch.randn(1, 3, 544, 960)
pre_img = torch.randn(1, 3, 544, 960)
pre_hm = torch.randn(1, 1, 544, 960)

output1 = backbone(x, pre_img, pre_hm)

output2 = backbone_ori(x, pre_img, pre_hm)
assert all([(v1 == v2).all() for v1, v2 in zip(output1, output2)])


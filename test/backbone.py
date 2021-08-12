import torch
from mmdet.models.backbones.dla import DLA

crowdhuman_path = '/home/akio/Downloads/new_crowdhuman.pth'

st = torch.load(crowdhuman_path)['state_dict']

backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512])


for k,v in backbone.state_dict().items():
    print(k,v.shape)
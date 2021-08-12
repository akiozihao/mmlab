import torch
from mmdet.models.backbones.dla import DLA
from mmdet.models.necks.dla_neck import DLANeck
from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLA as DLA_Ori
from CenterTrack.src.lib.model.networks.dla import DLASeg
from utils import Struct


# neck_pth = '/home/akio/Downloads/crowdhuman_split/neck.pt'
# opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'
neck_pth = '../tensors/neck.pt'
opt_path = '../tensors/opt.pt'

opt = torch.load(opt_path)
print(Struct.to_dict(opt))

neck = DLANeck(34).cuda()
heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}
head_convs = {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256], 'ltrb_amodal': [256]}
neck_ori = DLASeg(34, heads, head_convs, opt=opt)

x = torch.randn(1, 3, 544, 960).cuda()
pre_img = torch.randn(1, 3, 544, 960).cuda()
pre_hm = torch.randn(1, 1, 544, 960).cuda()

backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512]).cuda()
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1],
                       [16, 32, 64, 128, 256, 512],
                       block=BasicBlock)
backbone_out = backbone(x, pre_img, pre_hm)
backbone_out_ori = backbone_ori(x, pre_img, pre_hm)


neck_out = neck(backbone_out)  # tensor
x = neck_ori.dla_up(backbone_out_ori)
y = []
for i in range(neck_ori.last_level - neck_ori.first_level):
    y.append(x[i].clone())
neck_ori.ida_up(y, 0, len(y))
neck_out_ori = [y[-1]]
print('neck_out.shape',neck_out.shape)
print('done')

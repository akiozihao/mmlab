import torch

from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLASeg
from CenterTrack.src.lib.model.networks.dla import DLA as DLA_Ori
from mmdet.models.backbones.dla import DLA
from mmdet.models.necks.dla_neck import DLANeck

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
use_cuda = True
backbone_path_ori = '../mmpth/test_o_backbone.pt'
backbone_path = '../mmpth/test_mm_backbone.pt'
neck_path = '../mmpth/test_mm_neck.pt'
neck_path_ori = '../mmpth/test_o_neck.pt'
opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'

opt = torch.load(opt_path)

heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}
head_convs = {
    'hm': [256],
    'reg': [256],
    'wh': [256],
    'tracking': [256],
    'ltrb_amodal': [256]
}

x = torch.randn(1, 3, 544, 960)
pre_img = torch.randn(1, 3, 544, 960)
pre_hm = torch.randn(1, 1, 544, 960)

backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256,512],
               norm_cfg=dict(type='BN', momentum=0.1))
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                       block=BasicBlock,
                       opt=opt)
neck = DLANeck(channels=[16, 32, 64, 128, 256, 512],
               down_ratio=4,
               use_dcn=True,
               norm_cfg=dict(type='BN', momentum=0.1),
               use_origin_dcn=True)
neck_ori = DLASeg(34, heads, head_convs, opt=opt)

backbone_st = torch.load(backbone_path)
backbone_st_ori = torch.load(backbone_path_ori)
backbone.load_state_dict(backbone_st)
backbone_ori.load_state_dict(backbone_st_ori)

if use_cuda:
    backbone = backbone.cuda()
    backbone_ori = backbone_ori.cuda()
    neck = neck.cuda()
    neck_ori = neck_ori.cuda()
    x = x.cuda()
    pre_img = pre_img.cuda()
    pre_hm = pre_hm.cuda()

backbone_out = backbone(x, pre_img, pre_hm)
backbone_out_ori = backbone_ori(x, pre_img, pre_hm)

assert all([(v1 == v2).all() for v1, v2 in zip(backbone_out, backbone_out_ori)
            ]), 'backbone != backbone_ori'

neck_st = torch.load(neck_path)
neck_ori_st = torch.load(neck_path_ori)
neck.load_state_dict(neck_st)
neck_ori.load_state_dict(neck_ori_st,strict=False)
# origin partial seg
x_ori = neck_ori.dla_up(backbone_out_ori)
y_ori = []
for i in range(neck_ori.last_level - neck_ori.first_level):
    y_ori.append(x_ori[i].clone())
neck_ori.ida_up(y_ori, 0, len(y_ori))
neck_out_ori = [y_ori[-1]]
# ---------------------
neck_out = neck(backbone_out)
assert (neck_out[-1] == neck_out_ori[-1]).all(), 'neck_out != neck_out_ori'

print('done')

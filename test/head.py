import torch

from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLASeg
from CenterTrack.src.lib.model.networks.dla import DLA as DLA_Ori
from mmdet.models.backbones.dla import DLA
from mmdet.models.dense_heads.centertrack_head import CenterTrackHead
from mmdet.models.necks.dla_neck import DLANeck

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
use_cuda = True
backbone_path_ori = '../mmpth/test_o_backbone.pt'
backbone_path = '../mmpth/test_mm_backbone.pt'
neck_path = '../mmpth/test_mm_neck.pt'
neck_path_ori = '../mmpth/test_o_neck.pt'
head_path = '../mmpth/test_mm_head.pt'
head_path_ori = '../mmpth/test_o_head.pt'
opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'

heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}
head_convs = {
    'hm': [256],
    'reg': [256],
    'wh': [256],
    'tracking': [256],
    'ltrb_amodal': [256]
}


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _sigmoid_output(output):
    if 'hm' in output:
        output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
        output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output


opt = torch.load(opt_path)

# input
x = torch.randn(1, 3, 544, 960)
pre_img = torch.randn(1, 3, 544, 960)
pre_hm = torch.randn(1, 1, 544, 960)
# init backbone
backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512],
               norm_cfg=dict(type='BN', momentum=0.1))
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                       block=BasicBlock,
                       opt=opt)
# init neck
neck = DLANeck(channels=[16, 32, 64, 128, 256, 512],
               down_ratio=4,
               use_dcn=True,
               norm_cfg=dict(type='BN', momentum=0.1),
               use_origin_dcn=True)
neck_ori = DLASeg(34, heads, head_convs, opt=opt)
# init head

head = CenterTrackHead(in_channel=64,
                       feat_channel=256,
                       use_origin_gaussian_radius=True,
                       train_cfg=dict(fp_disturb=0.1, lost_disturb=0.4,
                                      hm_disturb=0.05)
                       )
# init origin model
seg = DLASeg(34, heads, head_convs, opt=opt)

# load backbone state_dict
backbone_st = torch.load(backbone_path)
backbone_st_ori = torch.load(backbone_path_ori)
backbone.load_state_dict(backbone_st)
backbone_ori.load_state_dict(backbone_st_ori)
# load neck state_dict
neck_st = torch.load(neck_path)
neck_st_ori = torch.load(neck_path_ori)
neck.load_state_dict(neck_st)
seg.load_state_dict(neck_st_ori, strict=False)
# load head state_dict
head_st = torch.load(head_path)
head_st_ori = torch.load(head_path_ori)
head.load_state_dict(head_st)
seg.load_state_dict(head_st_ori, strict=False)
# move to cuda
if use_cuda:
    backbone = backbone.cuda()
    backbone_ori = backbone_ori.cuda()
    neck = neck.cuda()
    seg = seg.cuda()
    head = head.cuda()
    x = x.cuda()
    pre_img = pre_img.cuda()
    pre_hm = pre_hm.cuda()

# backbone forward
backbone_out = backbone(x, pre_img, pre_hm)
backbone_out_ori = backbone_ori(x, pre_img, pre_hm)

assert all([(v1 == v2).all() for v1, v2 in zip(backbone_out, backbone_out_ori)
            ]), 'backbone != backbone_ori'
# neck forward
# origin partial seg
x_ori = seg.dla_up(backbone_out_ori)
y_ori = []
for i in range(seg.last_level - seg.first_level):
    y_ori.append(x_ori[i].clone())
seg.ida_up(y_ori, 0, len(y_ori))
neck_out_ori = y_ori[-1]
# ---------------------
neck_out = neck(backbone_out)
assert (neck_out[-1] == neck_out_ori[-1]).all(), 'neck_out != neck_out_ori'

# head forward
head_output = head(neck_out)

head_output_ori = {}
for head in seg.heads:
    head_output_ori[head] = seg.__getattr__(head)(neck_out_ori)
head_output_ori = [head_output_ori]
head_output_ori[0] = _sigmoid_output(head_output_ori[0])
name = dict()
name['hm'] = 'center_heatmap_pred'
name['reg'] ='offset_pred'
name['wh'] = 'wh_pred'
name['tracking'] = 'tracking_pred'
name['ltrb_amodal'] = 'ltrb_amodal_pred'
for head in seg.heads:
    assert (head_output[name[head]] == head_output_ori[0][head]
            ).all(), f'{head} not match'
print('done head')

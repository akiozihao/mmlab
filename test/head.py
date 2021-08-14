import torch
from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLA as DLA_Ori
from CenterTrack.src.lib.model.networks.dla import DLASeg
from mmdet.models.backbones.dla import DLA
from mmdet.models.dense_heads.centertrack_head import CenterTrackHead
from mmdet.models.necks.dla_neck import DLANeck

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
use_cuda = True
backbone_path = '../mmpth/backbone.pt'
backbone_path_ori = '/home/akio/Downloads/crowdhuman_split/backbone.pt'
neck_path = '../mmpth/neck.pt'
neck_path_ori = '/home/akio/Downloads/crowdhuman_split/neck.pt'
opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'
head_path = '/home/akio/Downloads/crowdhuman_split/head.pt'
# opt = Struct(**{'pre_img': True,
#                 'pre_hm': True,
#                 'head_kernel': 3,
#                 'prior_bias': -4.6,
#                 'dla_node': 'dcn',
#                 'load_model': ''}
#              )
opt = torch.load(opt_path)

# input
x = torch.randn(1, 3, 544, 960)
pre_img = torch.randn(1, 3, 544, 960)
pre_hm = torch.randn(1, 1, 544, 960)
# init backbone
backbone = DLA(levels=[1, 1, 1, 2, 2, 1],
               channels=[16, 32, 64, 128, 256, 512])
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1],
                       [16, 32, 64, 128, 256, 512],
                       block=BasicBlock, opt=opt)
# init neck
neck = DLANeck(channels=[16, 32, 64, 128, 256, 512],
               down_ratio=4)
# init head
head_convs = {
    'hm': [256],
    'reg': [256],
    'wh': [256],
    'tracking': [256],
    'ltrb_amodal': [256]
}
heads = {
    'hm': 1,
    'reg': 2,
    'wh': 2,
    'tracking': 2,
    'ltrb_amodal': 4
}

head = CenterTrackHead(
    heads, head_convs, 1, 64,weights=dict(hm=1, reg=1, wh=0.1, tracking=1, ltrb_amodal=0.1)
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
head.load_state_dict(head_st)
seg.load_state_dict(head_st, strict=False)
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

assert all([(v1 == v2).all() for v1, v2 in zip(backbone_out, backbone_out_ori)]), 'backbone != backbone_ori'
# neck forward
## origin partial seg
x_ori = seg.dla_up(backbone_out_ori)
y_ori = []
for i in range(seg.last_level - seg.first_level):
    y_ori.append(x_ori[i].clone())
seg.ida_up(y_ori, 0, len(y_ori))
neck_out_ori = y_ori[-1]
## ---------------------
neck_out = neck(backbone_out)
assert (neck_out[-1] == neck_out_ori[-1]).all(), 'neck_out != neck_out_ori'

# head forward
head_output = head(neck_out)

head_output_ori = {}
for head in seg.heads:
    head_output_ori[head] = seg.__getattr__(head)(neck_out_ori)
head_output_ori = [head_output_ori]
for head in seg.heads:
    assert (head_output[0][head] == head_output_ori[0][head]).all(), f'{head} not match'
print('done')

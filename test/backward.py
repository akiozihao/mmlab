import torch
from mmdet.models.backbones.dla import DLA
from mmdet.models.dense_heads.centertrack_head import CenterTrackHead
from mmdet.models.necks.dla_neck import DLANeck

# origin loss
from CenterTrack.src.lib.model.losses import FastFocalLoss, RegWeightedL1Loss
from CenterTrack.src.lib.model.networks.dla import DLA as DLA_Ori
from CenterTrack.src.lib.model.networks.dla import BasicBlock, DLASeg
from CenterTrack.src.lib.model.utils import _sigmoid

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

use_cuda = False
backbone_path = '../mmpth/backbone.pt'
backbone_path_ori = '/home/akio/Downloads/crowdhuman_split/backbone.pt'
neck_path = '../mmpth/neck.pt'
neck_path_ori = '/home/akio/Downloads/crowdhuman_split/neck.pt'
opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'
head_path = '/home/akio/Downloads/crowdhuman_split/head.pt'
batch_path = '/home/akio/Downloads/crowdhuman_split/batch.pt'
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
backbone = DLA(levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512])
backbone_ori = DLA_Ori([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                       block=BasicBlock,
                       opt=opt)
# init neck
neck = DLANeck(channels=[16, 32, 64, 128, 256, 512],
               down_ratio=4,
               use_dcn=True)

# init head
head_convs = {
    'hm': [256],
    'reg': [256],
    'wh': [256],
    'tracking': [256],
    'ltrb_amodal': [256]
}
heads = {'hm': 1, 'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb_amodal': 4}

head = CenterTrackHead(heads=dict(hm=1, reg=2, wh=2, tracking=2,
                                  ltrb_amodal=4),
                       head_convs=dict(hm=[256],
                                       reg=[256],
                                       wh=[256],
                                       tracking=[256],
                                       ltrb_amodal=[256]),
                       num_stacks=1,
                       last_channel=64,
                       weights=dict(hm=1,
                                    reg=1,
                                    wh=0.1,
                                    tracking=1,
                                    ltrb_amodal=0.1),
                       test_cfg=dict(topk=100,
                                     local_maximum_kernel=3,
                                     max_per_img=100),
                       train_cfg=dict(fp_disturb=0.1,
                                      lost_disturb=0.4,
                                      hm_disturb=0.05))
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
for head_name in seg.heads:
    head_output_ori[head_name] = seg.__getattr__(head_name)(neck_out_ori)
head_output_ori = [head_output_ori]
for head_name in seg.heads:
    assert (head_output[0][head_name] == head_output_ori[0][head_name]
            ).all(), f'{head_name} not match'

# loss
batch = torch.load(batch_path)
if not use_cuda:
    for k, v in batch.items():
        batch[k] = v.cpu()
loss_out = head.loss(head_output, batch)


class GenericLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss(opt=opt)
        self.crit_reg = RegWeightedL1Loss()

        self.opt = opt

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}

        for s in range(opt.num_stacks):
            output = outputs[s]
            output = self._sigmoid_output(output)

            if 'hm' in output:
                losses['hm'] += self.crit(output['hm'], batch['hm'],
                                          batch['ind'], batch['mask'],
                                          batch['cat']) / opt.num_stacks

            regression_heads = [
                'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps', 'dep',
                'dim', 'amodel_offset', 'velocity'
            ]

            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'], batch['ind'],
                        batch[head]) / opt.num_stacks

            if 'hm_hp' in output:
                losses['hm_hp'] += self.crit(
                    output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                    batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
                if 'hp_offset' in output:
                    losses['hp_offset'] += self.crit_reg(
                        output['hp_offset'], batch['hp_offset_mask'],
                        batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

            if 'rot' in output:
                losses['rot'] += self.crit_rot(
                    output['rot'], batch['rot_mask'], batch['ind'],
                    batch['rotbin'], batch['rotres']) / opt.num_stacks

            if 'nuscenes_att' in output:
                losses['nuscenes_att'] += self.crit_nuscenes_att(
                    output['nuscenes_att'], batch['nuscenes_att_mask'],
                    batch['ind'], batch['nuscenes_att']) / opt.num_stacks

        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += opt.weights[head] * losses[head]

        return losses['tot'], losses


loss_ori = GenericLoss(opt=opt)
if use_cuda:
    loss_ori = loss_ori.cuda()

loss_total_ori, loss_out_ori = loss_ori(head_output_ori, batch)
for k in loss_out.keys():
    assert (loss_out[k] == (opt.weights)[k[5:]] *
            loss_out_ori[k[5:]]).all(), f'Loss: {k} not match'

loss_total = loss_total_ori.new_zeros(1)
for k, v in loss_out.items():
    loss_total += v
# backward
loss_total_ori.backward()
loss_total[0].backward()

for (name, param), (name_ori,
                    param_ori) in zip(backbone.named_parameters(),
                                      backbone_ori.named_parameters()):
    # assert name == name_ori
    if param.grad is not None:
        print(torch.abs((param.grad - param_ori.grad)).sum(), f'{name}')

print('done')

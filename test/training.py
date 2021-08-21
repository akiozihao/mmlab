import torch
from mmcv.utils import import_modules_from_strings
from mmdet.models.detectors.ct_detector import CTDetector
from tqdm import tqdm

from CenterTrack.src.lib.dataset.datasets.mot import MOT

import_modules_from_strings(
    ['mmdet.models.backbones.dla',
     'mmdet.models.necks.dla_neck',
     'mmdet.models.dense_heads.centertrack_head',
     'mmdet.models.detectors.ct_detector',
     'mmtrack.models.mot.trackers.ct_tracker',
     'mmtrack.models.mot.center_track',
     'mmtrack.datasets.pipelines.transforms']
)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

backbone_path = '../mmpth/backbone.pt'
backbone_path_ori = '/home/akio/Downloads/crowdhuman_split/backbone.pt'
neck_path = '../mmpth/neck.pt'
neck_path_ori = '/home/akio/Downloads/crowdhuman_split/neck.pt'
opt_path = '/home/akio/Downloads/crowdhuman_split/opt.pt'
head_path = '/home/akio/Downloads/crowdhuman_split/head.pt'
batch_path = '/home/akio/Downloads/crowdhuman_split/batch.pt'

model = CTDetector(
    backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        norm_cfg=dict(type='BN', momentum=0.1),
    ),
    neck=dict(
        type='DLANeck',
        channels=[16, 32, 64, 128, 256, 512],
        down_ratio=4,
        use_dcn=True,
    ),
    bbox_head=dict(
        type='CenterTrackHead',
        heads=dict(hm=1, reg=2, wh=2, tracking=2, ltrb_amodal=4),
        head_convs=dict(hm=[256], reg=[256], wh=[256], tracking=[256], ltrb_amodal=[256]),
        num_stacks=1,
        last_channel=64,
        weights=dict(hm=1, reg=1, wh=0.1, tracking=1, ltrb_amodal=0.1)
    ),
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100),
    train_cfg=dict(fp_disturb=0.1, lost_disturb=0.4, hm_disturb=0.05)
)
# backbone_st = torch.load(backbone_path)
# neck_st = torch.load(neck_path)
# head_st = torch.load(head_path)
#
# model.backbone.load_state_dict(backbone_st)
# model.neck.load_state_dict(neck_st)
# model.bbox_head.load_state_dict(head_st)

st = torch.load('/home/akio/dev/mmlab/models/new_model.pth')['state_dict']
model.load_state_dict(st)
model = model.cuda()

opt = torch.load(opt_path)
opt.data_dir = '../data'

optimizer = torch.optim.Adam(model.parameters(), lr=1.25e-4 / 8)
train_loader = torch.utils.data.DataLoader(
    MOT(opt, 'train'), batch_size=2, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)
model.train()
total_epoch = 72
for epoch in range(1,total_epoch):
    print(f'Epoch {epoch} start.....')
    stat = []
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1.25e-5 / 8
    for batch in tqdm(train_loader):
        for k, v in batch.items():
            batch[k] = v.cuda()
        _, loss, loss_stats = model.forward_train(batch)
        loss = loss.mean()
        for k, v in loss_stats.items():
            loss += v
        stat.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} mean loss : {torch.tensor(stat).mean()}')
    stat.clear()

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'/home/akio/dev/mmlab/models/4_epoch_{epoch}.pth')
        break

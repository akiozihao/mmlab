_base_ = ['./centertrack_dla34_dcnv2_70e_mot17_private-half.py']
crowdhuman_data_root = '../data/crowdhuman/'
img_norm_cfg = dict(
    mean=[104.01362, 114.034225, 119.916595],
    std=[73.60277, 69.89082, 70.91508],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCenterAffine',
        crop_size=(512, 512),
        ratios=None,
        shift=0.05,
        scale=0.05,
        share_params=False),
    dict(
        type='SeqResize',
        img_scale=(512, 512),
        share_params=True,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]

model = dict(
    init_cfg=None,
    detector=dict(
        backbone=dict(
            init_cfg=dict(
                type='Pretrained',
                checkpoint='../models/mmlab_dla34-ba72cf86.pth'))))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            ann_file=crowdhuman_data_root +
            'annotations/train_cocoformat.json',
            img_prefix=crowdhuman_data_root + 'Images',
            ref_img_sampler=dict(
                frame_range=0,
                filter_key_img=False,
            ),
            pipeline=train_pipeline)),
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.01,  # centernet 5e-4 64bs use 0.02
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[90, 120])

total_epochs = 140

checkpoint_config = dict(interval=20)

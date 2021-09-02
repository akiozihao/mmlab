custom_imports = dict(
    imports=['mmdet.models.backbones.dla',
             'mmdet.models.necks.dla_neck',
             'mmdet.models.dense_heads.centertrack_head',
             'mmdet.models.detectors.ct_detector',
             'mmtrack.models.mot.trackers.ct_tracker',
             'mmtrack.models.mot.center_track',
             'mmtrack.datasets.pipelines.transforms'],
    allow_failed_imports=False
)
_base_ = [
    '../../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'MOTChallengeDataset'
img_norm_cfg = dict(
    mean=[104.01362, 114.034225, 119.916595], std=[73.60277, 69.89082, 70.91508], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCenterAffine',
        crop_size=(544, 960),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)),
    dict(
        type='SeqResize',
        img_scale=(544, 960),
        share_params=True,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]
    ),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(544, 960)],
        # scale_factor=1.0,
        flip=False,
        transforms=[
            dict(
                type='SeqRandomCenterAffine',
                test_mode=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='VideoCollect',
                meta_keys=('pad_shape', 'invert_transform'),
                keys=['img'])
        ]
    )
]

data_root = '../data/mot17-frcnn/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            visibility_thr=0.25,
            ann_file=data_root + 'annotations/half-train_cocoformat.json',
            img_prefix=data_root + 'train',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=2,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline)
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline
    )
)
model = dict(
    type='CenterTrack',
    init_cfg=dict(type='Pretrained', checkpoint='../models/mmlab_crowdhuman.pth'),  # here for mmlab checkpoints
    detector=dict(
        type='CTDetector',
        # init_cfg=dict(type='Pretrained', checkpoint='../models/new_crowdhuman_mmdcn_head_neck.pth'),  # here for original models
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
            in_channel=64,
            feat_channel=256,
        ),
        test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100),
        train_cfg=dict(fp_disturb=0.1, lost_disturb=0.4, hm_disturb=0.05),
        use_origin_gaussian_radius=False,
    ),
    tracker=dict(type='CTTracker',
                 obj_score_thr=0.4,
                 num_frames_retain=3,
                 momentums=dict(
                     ids=1,
                     bboxes_input=1,
                     bboxes=1,
                     cts=1,
                     labels=1,
                     frame_ids=1,
                 ))
)

# optimizer
optimizer = dict(_delete_=True, type='Adam', lr=1.25e-4 / 2)
# optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=1000,
    # warmup_ratio=1.0 / 1000,
    step=[60]
)

# runtime settings
total_epochs = 70
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# For distributed training
find_unused_parameters = True

# checkpoint
checkpoint_config = dict(_delete_=True, interval=10)

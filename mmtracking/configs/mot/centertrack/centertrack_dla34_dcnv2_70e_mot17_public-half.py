_base_ = ['./centertrack_dla34_dcnv2_70e_mot17_private-half.py']
img_norm_cfg = dict(
    mean=[104.01362, 114.034225, 119.916595], std=[73.60277, 69.89082, 70.91508], to_rgb=False
)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadDetections'),
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
                keys=['img', 'public_bboxes', 'public_labels', 'public_scores'])
        ]
    )
]
data_root = '../data/mot17-frcnn/'
data = dict(
    val=dict(ann_file=data_root + 'annotations/half-val_cocoformat.json',
             img_prefix=data_root + 'train',
             detection_file=data_root + 'annotations/half-val_detections.pkl',
             pipeline=test_pipeline),
    test=dict(ann_file=data_root + 'annotations/half-val_cocoformat.json',
              img_prefix=data_root + 'train',
              detection_file=data_root + 'annotations/half-val_detections.pkl',
              pipeline=test_pipeline)
)

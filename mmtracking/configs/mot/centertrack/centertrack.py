custom_imports = dict(
    imports=['mmdet.models.backbones.dla',
             'mmdet.models.neck.dla_neck',
             'mmdet.models.dense_heads.centertrack_head'],
    allow_failed_imports=False
)

model = dict(
    type='CenterTrack',
    detector=dict(
        type='CTDetector',
        backbone=dict(
            type='DLA',
            levels=[1, 1, 1, 2, 2, 1],
            channels=[16, 32, 64, 128, 256, 512],
        ),
        neck=dict(
            type='DLANeck',
            arch=34),
    )
)

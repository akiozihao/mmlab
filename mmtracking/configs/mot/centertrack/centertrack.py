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
            norm_cfg=dict(type='BN', momentum=0.1),
        ),
        neck=dict(
            type='DLANeck',
            channels=[16, 32, 64, 128, 256, 512],
            down_ratio=4,
            use_dcn=True,
        ),
        head=dict(
            type='CenterTrackHead',
            heads=dict(hm=1, reg=2, wh=2, tracking=2, ltrb_amodal=4),
            head_convs=dict(hm=[256], reg=[256], wh=[256], tracking=[256], ltrb_amodal=[256]),
            num_statcks=1,
            last_channel=64,
            weight=dict(hm=1, reg=1, wh=0.1, tracking=1, ltrb_amodal=0.1)
        )
    )
)

USE_MMCLS = True
_base_ = ['./resnet50_b32x8_MOT17.py']
model = dict(reid=dict(head=dict(num_classes=375)))
# data
data_root = 'data/MOT16/'
data = dict(
    train=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/train_80.txt'),
    val=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'),
    test=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'))

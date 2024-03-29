_base_ = ['./centertrack_dla34_dcnv2_70e_mot17_public-half.py']
data_root = '../data/mot17-frcnn/'
data = dict(
    val=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/train_detections.pkl'),
    test=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/train_detections.pkl'))

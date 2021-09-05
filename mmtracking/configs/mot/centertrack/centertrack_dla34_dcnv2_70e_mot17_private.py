_base_ = ['./centertrack_dla34_dcnv2_70e_mot17_private-half.py']
data_root = '../data/mot17-frcnn/'
data = dict(
    train=dict(
        dataset=dict(ann_file=data_root +
                     'annotations/train_cocoformat.json')),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))

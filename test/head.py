import torch
from mmdet.models.dense_heads.centertrack_head import CenterTrackHead

head_convs = {
    'hm': [256],
    'reg': [256],
    'wh': [256],
    'tracking': [256],
    'ltrb_amodal': [256]
}
heads = {
    'hm': 1,
    'reg': 2,
    'wh': 2,
    'tracking': 2,
    'ltrb_amodal': 4
}

head = CenterTrackHead(
    heads, head_convs, 1, 64
)

head_input = [torch.randn(1, 64, 136, 240)]

head_output = head(head_input)

print('done')

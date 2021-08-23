import torch
from mmcv import ConfigDict
from mmdet.core import bbox2result
from mmdet.models.dense_heads.centertrack_head import CenterTrackHead
from mmtrack.core import track2result

from detector import Detector
from mmtrack.models.mot.trackers.ct_tracker import CTTracker
from model.decode import generic_decode
from utils.image import get_affine_transform

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
use_cuda = True

# Load tensors
meta_path = '../tensors/meta.pt'
opt_path = '../tensors/opt.pt'
output1_path = '../tensors/head_output_1.pt'
output2_path = '../tensors/head_output_1.pt'
meta = torch.load(meta_path)
opt = torch.load(opt_path)
opt.out_thresh = 0.4
output1 = torch.load(output1_path)
output2 = torch.load(output2_path)

# Init modules
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
    heads=dict(hm=1, reg=2, wh=2, tracking=2, ltrb_amodal=4),
    head_convs=dict(hm=[256], reg=[256], wh=[256], tracking=[256], ltrb_amodal=[256]),
    num_stacks=1,
    last_channel=64,
    weights=dict(hm=1, reg=1, wh=0.1, tracking=1, ltrb_amodal=0.1),
    test_cfg=ConfigDict(topk=100, local_maximum_kernel=3, max_per_img=100),
    train_cfg=ConfigDict(fp_disturb=0.1, lost_disturb=0.4, hm_disturb=0.05)
)
tracker = CTTracker(obj_score_thr=0.4, momentums=None, num_frames_retain=1)
# original
detector = Detector(opt)

# Get bbox from head outputs
center_heatmap_pred = output1['hm']
wh_pred = output1['wh']
offset_pred = output1['reg']
tracking_pred = output1['tracking']
ltrb_amodal_pred = output1['ltrb_amodal']
outs = [center_heatmap_pred, wh_pred, offset_pred, tracking_pred, ltrb_amodal_pred]
trans = get_affine_transform(meta['c'], meta['s'], 0, [meta['inp_width'], meta['inp_height']], inv=1)
img_metas = [{'invert_transform': trans, 'batch_input_shape': [meta['inp_height'], meta['inp_width']]} for _ in center_heatmap_pred]
result_list = head.get_bboxes(
    *[[tensor] for tensor in outs], img_metas=img_metas)
# original
dets = generic_decode(output1, K=opt.K, opt=opt)
for k in dets:
    dets[k] = dets[k].detach().cpu().numpy()
result = detector.post_process(dets, meta, scale=1)
results = detector.merge_outputs([result])

# Track
frame_id = 0
det_bboxes, det_labels, det_bboxes_with_motion, det_bboxes_input = result_list[0]
bboxes, labels, ids = tracker.track(
            bboxes_input=det_bboxes_input,
            bboxes=det_bboxes,
            bboxes_with_motion=det_bboxes_with_motion,
            labels=det_labels,
            frame_id=frame_id)
num_classes = 1
track_result = track2result(bboxes, labels, ids, num_classes)
bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
# original
results = detector.tracker.step(results, None)

print('done track')

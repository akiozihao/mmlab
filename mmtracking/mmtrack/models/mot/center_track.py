import typing

import torch
from mmdet.core import bbox2result
from mmdet.models import build_detector

from mmtrack.core import track2result
from ..builder import MODELS, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class CenterTrack(BaseMultiObjectTracker):
    """Implementation of CenterTrack (Tracking Objects as Points)

    Details can be found at 'CenterTrack<https://arxiv.org/abs/2004.01177>'.
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 pre_thresh=0.5,
                 use_pre_hm=True,
                 init_cfg=None):
        super(CenterTrack, self).__init__(init_cfg)
        self.detector = build_detector(detector)
        self.tracker = build_tracker(tracker)

        self.pre_thresh = pre_thresh
        self.use_pre_hm = use_pre_hm

        self.ref_img = None
        self.ref_hm = None

    def simple_test(self,
                    img,
                    img_metas,
                    public_bboxes=None,
                    public_labels=None,
                    public_scores=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (torch.Tensor): shape (N, C, H, W) Encoded input images.
            img_metas (list[dict]): List of image info.
            public_bboxes (list[torch.Tensor], optional): Public bounding
                boxes from the benchmark. Defaults to None.
            public_labels (list[torch.Tensor], optional): Labels of public
                bounding  boxes from the benchmark. Defaults to None.
            public_scores (list[torch.Tensor], optional): Scores of public
                bounding boxes from the benchmark. Defaults to None.

        Returns:
            dict[str, list[numpy.ndarray]]: The tracking results.
        """
        frame_id = img_metas[0]['frame_id']
        pre_active_bboxes_input = self.tracker.pre_active_bboxes_input(
            frame_id)
        if pre_active_bboxes_input is not None:
            pre_active_bboxes_input = pre_active_bboxes_input[
                pre_active_bboxes_input[:, -1] > self.pre_thresh]
        if frame_id == 0:
            self.tracker.reset()
            self.ref_img = img.clone()
            if self.use_pre_hm:
                n, c, h, w = img.shape
                self.ref_hm = torch.zeros((n, 1, h, w),
                                          dtype=img.dtype,
                                          device=img.device)
            else:
                self.ref_hm = None
                self.ref_img = None
        else:
            if self.use_pre_hm:
                if pre_active_bboxes_input is None or \
                        pre_active_bboxes_input.shape[0] == 0:
                    n, c, h, w = img.shape
                    self.ref_hm = torch.zeros((n, 1, h, w),
                                              dtype=img.dtype,
                                              device=img.device)
                else:
                    self.ref_hm = self.detector._build_test_hm(
                        self.ref_img, pre_active_bboxes_input)
            else:
                self.ref_hm = None
                self.ref_img = None

        batch_input_shape = tuple(img[0].size()[-2:])
        img_metas[0]['batch_input_shape'] = batch_input_shape
        x = self.detector.extract_feat(img, self.ref_img, self.ref_hm)
        bbox_head_out = self.detector.bbox_head(x)
        center_heatmap_pred = bbox_head_out['center_heatmap_pred']
        wh_pred = bbox_head_out['wh_pred']
        offset_pred = bbox_head_out['offset_pred']
        tracking_pred = bbox_head_out['tracking_pred']
        ltrb_amodal_pred = bbox_head_out['ltrb_amodal_pred']
        outs = [
            center_heatmap_pred, wh_pred, offset_pred, tracking_pred,
            ltrb_amodal_pred
        ]
        result_list = self.detector.bbox_head.get_bboxes(
            # todo Are outs always tensors?
            *[[tensor] for tensor in outs],
            img_metas=img_metas)
        # TODO: support batch inference.
        det_bboxes = result_list[0][0]
        det_labels = result_list[0][1]
        det_bboxes_input = result_list[0][2]
        det_centers = result_list[0][3]
        det_tracking_offset = result_list[0][4]
        num_classes = self.detector.bbox_head.num_classes
        self.ref_img = img
        # reformat public dets
        if public_bboxes is not None:
            assert public_labels is not None and public_scores is not None
            public_labels = public_labels[0][0]
            public_bboxes = torch.cat(
                (public_bboxes[0][0], public_scores[0][0].unsqueeze(-1)), -1)
        bboxes, labels, ids = self.tracker.track(
            bboxes_input=det_bboxes_input,
            bboxes=det_bboxes,
            det_centers=det_centers,
            det_tracking_offset=det_tracking_offset,
            labels=det_labels,
            frame_id=frame_id,
            public_bboxes=public_bboxes,
            public_labels=public_labels)
        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_instance_ids, gt_match_indices, ref_img_metas,
                      ref_img, ref_gt_bboxes, ref_gt_labels,
                      ref_gt_match_indices, ref_gt_instance_ids):
        """Forward function during training.

        We directly use the forward_train function of ct_detector. Amodal means
        that the bboxes can be outside of the image.
        """
        gt_amodal_bboxes = gt_bboxes
        ref_amodal_bboxes = ref_gt_bboxes
        return self.detector.forward_train(
            img,
            img_metas,
            gt_amodal_bboxes,
            gt_labels,
            gt_match_indices,
            ref_img,
            ref_amodal_bboxes,
        )

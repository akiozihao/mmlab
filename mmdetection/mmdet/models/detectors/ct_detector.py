import random

import torch
from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS
from mmdet.models.utils import gen_gaussian_target, gaussian_radius

from .centernet import CenterNet


@DETECTORS.register_module()
class CTDetector(CenterNet):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(CTDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)
        self.fp_disturb = 0.1
        self.hm_disturb = 0.05
        self.lost_disturb = 0.4

    @auto_fp16()
    def extract_feat(self, img, pre_img, pre_hm):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, pre_img, pre_hm)
        x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img_metas,
                      ref_img,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices):
        ref_hm = self._build_ref_hm(ref_gt_bboxes, ref_gt_labels, img.shape)
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        for img_meta in ref_img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.backbone(img, ref_img, ref_hm)
        x = self.neck(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_match_indices,
                                              ref_img_metas, ref_gt_bboxes, ref_gt_labels, ref_gt_match_indices,
                                              ref_hm)
        return losses

    def _build_ref_hm(self, gt_bboxes, gt_labels, img_shape):
        batch_size, _, img_h, img_w = img_shape

        heatmap = gt_bboxes[-1].new_zeros([batch_size, self.bbox_head.num_classes, img_h, img_w])

        for batch_id in range(batch_size):
            gt_bbox = gt_bboxes[batch_id].clone()
            # clip
            gt_bbox[:, [0, 2]] = torch.clip(gt_bbox[:, [0, 2]], 0, img_w)
            gt_bbox[:, [1, 3]] = torch.clip(gt_bbox[:, [1, 3]], 0, img_h)
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                box_h = (gt_bbox[j][3] - gt_bbox[j][1])
                box_w = (gt_bbox[j][2] - gt_bbox[j][0])
                ctx_int, cty_int = ct.int()
                radius = gaussian_radius([box_h, box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(heatmap[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                if random.random() < self.fp_disturb:
                    ct2 = ct.clone()
                    ct2[0] = ct2[0] + random.random() * 0.05 * box_w
                    ct2[1] = ct2[1] + random.random() * 0.05 * box_h
                    ctx2_int, cty2_int = ct2.int()
                    gen_gaussian_target(heatmap[0, 0],
                                        [ctx2_int, cty2_int], radius)
        return heatmap

    def _build_test_hm(self, ref_img, ref_bboxes):
        batch_size, _, img_h, img_w = ref_img.shape
        assert batch_size == 1
        bboxes = ref_bboxes.clone()
        heatmap = bboxes.new_zeros([batch_size, 1, img_h, img_w])
        # clip
        bboxes[:, [0, 2]] = torch.clip(bboxes[:, [0, 2]], 0, img_w)
        bboxes[:, [1, 3]] = torch.clip(bboxes[:, [1, 3]], 0, img_h)
        center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        centers = torch.cat((center_x.reshape(-1, 1), center_y.reshape(-1, 1)), dim=1)

        for j, ct in enumerate(centers):
            scale_box_h = (bboxes[j][3] - bboxes[j][1])
            scale_box_w = (bboxes[j][2] - bboxes[j][0])
            if scale_box_h <= 0 or scale_box_w <= 0:
                continue
            radius = gaussian_radius([scale_box_h, scale_box_w],
                                     min_overlap=0.3)  # check
            radius = max(0, int(radius))

            ct[0] = torch.ceil(ct[0] + random.random() * self.bbox_head.hm_disturb * scale_box_w)
            ct[1] = torch.ceil(ct[1] + random.random() * self.bbox_head.hm_disturb * scale_box_h)
            ctx_int, cty_int = ct.int()
            gen_gaussian_target(heatmap[0, 0],
                                [ctx_int, cty_int], radius)
            # if random.random() < self.bbox_head.fp_disturb:
            #     ct2 = ct.clone()
            #     ct2[0] = ct2[0] + random.random() * 0.05 * scale_box_w
            #     ct2[1] = ct2[1] + random.random() * 0.05 * scale_box_h
            #     ctx2_int, cty2_int = ct2.int()
            #     gen_gaussian_target(heatmap[0, 0], [ctx2_int, cty2_int], radius)

        return heatmap
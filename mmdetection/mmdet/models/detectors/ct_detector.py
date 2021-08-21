import math
import random
from math import sqrt

import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.utils import gen_gaussian_target

from .single_stage import SingleStageDetector


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


@DETECTORS.register_module()
class CTDetector(SingleStageDetector):
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
        self.fp_disturb = train_cfg['fp_disturb']
        self.hm_disturb = train_cfg['hm_disturb']
        self.lost_disturb = train_cfg['lost_disturb']
        self.num_classes = 1

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
                      ref_img,
                      ref_gt_bboxes,
                      ref_gt_labels):
        batch = self._input2targets(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_match_indices,
            ref_img,
            ref_gt_bboxes,
            ref_gt_labels,
            down_ratio=4
        )
        img = batch['image']
        ref_img = batch['pre_img']
        ref_hm = batch['pre_hm']
        # img_metas = batch['img_metas']
        batch_input_shape = tuple(img[0].size()[-2:])

        # for img_meta in img_metas:
        #     img_meta['batch_input_shape'] = batch_input_shape
        x = self.backbone(img, ref_img, ref_hm)
        x = self.neck(x)
        losses = self.bbox_head.forward_train(x, batch)
        return losses

    # def _build_ref_hm(self, gt_bboxes, gt_labels, img_shape):
    #     batch_size, _, img_h, img_w = img_shape
    #
    #     heatmap = gt_bboxes[-1].new_zeros([batch_size, self.bbox_head.num_classes, img_h, img_w])
    #
    #     for batch_id in range(batch_size):
    #         gt_bbox = gt_bboxes[batch_id].clone()
    #         # clip
    #         gt_bbox[:, [0, 2]] = torch.clip(gt_bbox[:, [0, 2]], 0, img_w - 1)
    #         gt_bbox[:, [1, 3]] = torch.clip(gt_bbox[:, [1, 3]], 0, img_h - 1)
    #         gt_label = gt_labels[batch_id]
    #         center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
    #         center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
    #         gt_centers = torch.cat((center_x, center_y), dim=1)
    #
    #         for j, ct in enumerate(gt_centers):
    #             box_h = (gt_bbox[j][3] - gt_bbox[j][1])
    #             box_w = (gt_bbox[j][2] - gt_bbox[j][0])
    #             if box_w <= 0 or box_h <= 0:
    #                 continue
    #
    #             radius = gaussian_radius([torch.ceil(box_h), torch.ceil(box_w)])
    #             radius = max(0, int(radius))
    #             ind = gt_label[j]
    #
    #             ct[0] = ct[0] + random.random() * self.hm_disturb * box_w
    #             ct[1] = ct[1] + random.random() * self.hm_disturb * box_h
    #             ctx_int, cty_int = ct.int()
    #             gen_gaussian_target(heatmap[batch_id, ind],
    #                                 [ctx_int, cty_int], radius)
    #
    #             if random.random() < self.fp_disturb:
    #                 ct2 = ct.clone()
    #                 ct2[0] = ct2[0] + random.random() * 0.05 * box_w
    #                 ct2[1] = ct2[1] + random.random() * 0.05 * box_h
    #                 ctx2_int, cty2_int = ct2.int()
    #                 gen_gaussian_target(heatmap[0, 0],
    #                                     [ctx2_int, cty2_int], radius)
    #     return heatmap

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
            radius = gaussian_radius([scale_box_h, scale_box_w])  # check
            radius = max(0, int(radius))

            ct[0] = torch.ceil(ct[0] + torch.randn(1,device=bboxes.device) * self.bbox_head.hm_disturb * scale_box_w)
            ct[1] = torch.ceil(ct[1] + torch.randn(1,device=bboxes.device) * self.bbox_head.hm_disturb * scale_box_h)
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

    def _input2targets(self,
                       img,
                       img_metas,
                       gt_bboxes,
                       gt_labels,
                       gt_match_indices,
                       ref_img,
                       ref_gt_bboxes,
                       ref_gt_labels,
                       down_ratio=4):
        """
        Returns: dict
             img,ref_img,ref_hm
             hm_target,reg_target,tracking_target,wh_target,ltrb_amodal_target,ind
        """
        batch = dict()

        img_h, img_w = img_metas[0]['pad_shape'][:2]
        bs = img.shape[0]
        feat_w, feat_h = int(img_w / down_ratio), int(img_h / down_ratio)
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        max_obj = 256

        pre_hm = gt_bboxes[-1].new_zeros([bs, 1, img_h, img_w])

        hm = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        hm_mask = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        reg = gt_bboxes[-1].new_zeros([bs, max_obj, 2])
        reg_mask = gt_bboxes[-1].new_zeros([bs, max_obj, 2])

        tracking = gt_bboxes[-1].new_zeros([bs, max_obj, 2])
        tracking_mask = gt_bboxes[-1].new_zeros([bs, max_obj, 2])

        wh = gt_bboxes[-1].new_zeros([bs, max_obj, 2])
        wh_mask = gt_bboxes[-1].new_zeros([bs, max_obj, 2])

        ltrb_amodal = gt_bboxes[-1].new_zeros([bs, max_obj, 4])
        ltrb_amodal_mask = gt_bboxes[-1].new_zeros([bs, max_obj, 4])

        target_ind = gt_bboxes[-1].new_zeros([bs, max_obj], dtype=torch.int64)
        mask = gt_bboxes[-1].new_zeros([bs, max_obj])
        cat = gt_bboxes[-1].new_zeros([bs, max_obj], dtype=torch.int64)
        for batch_id in range(bs):
            # amodal scale gt bbox
            scale_gt_amodal_bbox = gt_bboxes[batch_id].clone()
            scale_gt_amodal_bbox[:, [0, 2]] = scale_gt_amodal_bbox[:, [0, 2]] * width_ratio
            scale_gt_amodal_bbox[:, [1, 3]] = scale_gt_amodal_bbox[:, [1, 3]] * height_ratio
            # clipped scale gt bbox
            scale_gt_bbox = gt_bboxes[batch_id].clone()
            scale_gt_bbox[:, [0, 2]] = torch.clip(scale_gt_bbox[:, [0, 2]] * width_ratio, 0, feat_w - 1)
            scale_gt_bbox[:, [1, 3]] = torch.clip(scale_gt_bbox[:, [1, 3]] * height_ratio, 0, feat_h - 1)
            #  clipped scale ref bbox
            scale_ref_bbox = ref_gt_bboxes[batch_id].clone()
            scale_ref_bbox[:, [0, 2]] = torch.clip(scale_ref_bbox[:, [0, 2]] * width_ratio, 0, feat_w - 1)
            scale_ref_bbox[:, [1, 3]] = torch.clip(scale_ref_bbox[:, [1, 3]] * height_ratio, 0, feat_h - 1)
            # clipped ref bbox
            ref_bbox = ref_gt_bboxes[batch_id].clone()
            ref_bbox[:, [0, 2]] = torch.clip(ref_bbox[:, [0, 2]], 0, img_w - 1)
            ref_bbox[:, [1, 3]] = torch.clip(ref_bbox[:, [1, 3]], 0, img_h - 1)
            # centers
            # amodal scale gt centers
            scale_gt_amodal_center_x = (scale_gt_amodal_bbox[:, [0]] + scale_gt_amodal_bbox[:, [2]]) / 2
            scale_gt_amodal_center_y = (scale_gt_amodal_bbox[:, [1]] + scale_gt_amodal_bbox[:, [3]]) / 2
            # clipped scale gt centers
            scale_gt_center_x = (scale_gt_bbox[:, [0]] + scale_gt_bbox[:, [2]]) / 2
            scale_gt_center_y = (scale_gt_bbox[:, [1]] + scale_gt_bbox[:, [3]]) / 2
            # clipped scale ref centers
            scale_ref_center_x = (scale_ref_bbox[:, [0]] + scale_ref_bbox[:, [2]]) / 2
            scale_ref_center_y = (scale_ref_bbox[:, [1]] + scale_ref_bbox[:, [3]]) / 2
            # clipped ref centers
            ref_center_x = (ref_bbox[:, [0]] + ref_bbox[:, [2]]) / 2
            ref_center_y = (ref_bbox[:, [1]] + ref_bbox[:, [3]]) / 2
            # labels
            gt_label = gt_labels[batch_id]

            # cat centers
            scale_gt_centers = torch.cat((scale_gt_center_x, scale_gt_center_y), dim=1)
            scale_ref_centers = torch.cat((scale_ref_center_x, scale_ref_center_y), dim=1)
            scale_gt_amodal_centers = torch.cat((scale_gt_amodal_center_x, scale_gt_amodal_center_y), dim=1)
            ref_centers = torch.cat((ref_center_x, ref_center_y), dim=1)
            # build ref hm and update ref_centers
            for idx in range(ref_centers.shape[0]):
                ref_h = ref_bbox[idx][3] - ref_bbox[idx][1]
                ref_w = ref_bbox[idx][2] - ref_bbox[idx][0]
                if ref_h <= 0 or ref_w <= 0:
                    continue

                radius = gaussian_radius((math.ceil(ref_h), math.ceil(ref_w)))
                radius = max(0, int(radius))

                ct0 = ref_centers[idx]
                ct = ref_centers[idx].clone()

                ct[0] = ct[0] + torch.randn(1,device=img.device) * self.hm_disturb * ref_w
                ct[1] = ct[1] + torch.randn(1,device=img.device) * self.hm_disturb * ref_h
                conf = 1 if torch.randn(1,device=img.device) > self.lost_disturb else 0

                ct_int = ct.int()
                if conf == 0:
                    ref_centers[idx] = ct

                gen_gaussian_target(pre_hm[batch_id, 0], ct_int, radius, k=conf)

                if torch.randn(1) < self.fp_disturb:
                    ct2 = ct0.clone()
                    ct2[0] = ct2[0] + torch.randn(1,device=img.device) * 0.05 * ref_w
                    ct2[1] = ct2[1] + torch.randn(1,device=img.device) * 0.05 * ref_h
                    ct2_int = ct2.int()
                    gen_gaussian_target(pre_hm[batch_id, 0], ct2_int, radius, k=conf)

            num_obj = min(max_obj, scale_gt_centers.shape[0])

            for j in range(num_obj):
                ct = scale_gt_centers[j]
                ctx, cty = ct
                scale_box_h = scale_gt_bbox[j][3] - scale_gt_bbox[j][1]
                scale_box_w = scale_gt_bbox[j][2] - scale_gt_bbox[j][0]
                if scale_box_h <= 0 or scale_box_w <= 0:
                    continue
                mask[batch_id, j] = 1
                cat[batch_id, j] = gt_label[j]
                radius = gaussian_radius([torch.ceil(scale_box_h), torch.ceil(scale_box_w)], min_overlap=0.3)
                radius = max(0, int(radius))

                ctx_int, cty_int = ct.int()
                gen_gaussian_target(hm[batch_id, gt_label[j]],
                                    [ctx_int, cty_int], radius)

                target_ind[batch_id, j] = cty_int * feat_w + ctx_int

                wh[batch_id, j, 0] = scale_box_w
                wh[batch_id, j, 1] = scale_box_h
                wh_mask[batch_id, j, :] = 1

                reg[batch_id, j, 0] = ctx - ctx_int
                reg[batch_id, j, 1] = cty - cty_int
                reg_mask[batch_id, j, :] = 1

                ltrb_amodal[batch_id, j, 0] = scale_gt_amodal_bbox[j, 0] - ctx_int
                ltrb_amodal[batch_id, j, 1] = scale_gt_amodal_bbox[j, 1] - cty_int
                ltrb_amodal[batch_id, j, 2] = scale_gt_amodal_bbox[j, 2] - ctx_int
                ltrb_amodal[batch_id, j, 3] = scale_gt_amodal_bbox[j, 3] - cty_int
                ltrb_amodal_mask[batch_id, j, :] = 1

                if gt_match_indices[batch_id][j] != -1:
                    idx = gt_match_indices[batch_id][j]
                    scale_ref_h = scale_ref_bbox[idx][3] - scale_ref_bbox[idx][1]
                    scale_ref_w = scale_ref_bbox[idx][2] - scale_ref_bbox[idx][0]
                    if scale_ref_h <= 0 or scale_ref_w <= 0:
                        continue
                    else:
                        scale_ref_ctx, scale_ref_cty = scale_ref_centers[idx]
                        tracking[batch_id, j, 0] = scale_ref_ctx - ctx_int
                        tracking[batch_id, j, 1] = scale_ref_cty - cty_int
                        tracking_mask[batch_id, j, :] = 1
        batch['image'] = img
        batch['pre_img'] = ref_img
        batch['pre_hm'] = pre_hm
        batch['hm'] = hm
        batch['hm_mask'] = hm_mask
        batch['wh'] = wh
        batch['wh_mask'] = wh_mask
        batch['reg'] = reg
        batch['reg_mask'] = reg_mask
        batch['ltrb_amodal'] = ltrb_amodal
        batch['ltrb_amodal_mask'] = ltrb_amodal_mask
        batch['tracking'] = tracking
        batch['tracking_mask'] = tracking_mask
        batch['mask'] = mask
        batch['cat'] = cat
        batch['img_metas'] = img_metas
        batch['ind'] = target_ind
        return batch

import torch

from mmdet.models.builder import DETECTORS
from mmdet.models.utils import gen_gaussian_target, gaussian_radius
from .single_stage import SingleStageDetector


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
                      gt_amodal_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_amodal_bboxes):
        ref_hm, ref_bboxes = self._build_ref_hm_update_ref_bboxes(ref_img, ref_amodal_bboxes)
        x = self.backbone(img, ref_img, ref_hm)
        x = self.neck(x)
        losses = self.bbox_head.forward_train(x, gt_amodal_bboxes, gt_labels, x.shape, img_metas[0]['pad_shape'],
                                              gt_match_indices, ref_bboxes)
        return losses

    def _build_ref_hm_update_ref_bboxes(self, ref_img, ref_bboxes):
        bs, _, img_h, img_w = ref_img.shape
        ref_hm = ref_bboxes[-1].new_zeros([bs, 1, img_h, img_w])
        for batch_id in range(bs):
            ref_bbox = ref_bboxes[batch_id]
            ref_bbox[:, [0, 2]] = torch.clip(ref_bbox[:, [0, 2]], 0, img_w - 1)
            ref_bbox[:, [1, 3]] = torch.clip(ref_bbox[:, [1, 3]], 0, img_h - 1)
            # clipped ref centers
            ref_center_x = (ref_bbox[:, [0]] + ref_bbox[:, [2]]) / 2
            ref_center_y = (ref_bbox[:, [1]] + ref_bbox[:, [3]]) / 2
            ref_centers = torch.cat((ref_center_x, ref_center_y), dim=1)
            # build ref hm and update ref_centers
            for idx in range(ref_centers.shape[0]):
                ref_h = ref_bbox[idx][3] - ref_bbox[idx][1]
                ref_w = ref_bbox[idx][2] - ref_bbox[idx][0]
                if ref_h <= 0 or ref_w <= 0:
                    continue

                radius = gaussian_radius([torch.ceil(ref_h), torch.ceil(ref_w)], min_overlap=0.3)
                radius = max(0, int(radius))

                ct0 = ref_centers[idx]
                ct = ref_centers[idx].clone()

                ct[0] = ct[0] + torch.randn(1, device=ref_img.device) * self.hm_disturb * ref_w
                ct[1] = ct[1] + torch.randn(1, device=ref_img.device) * self.hm_disturb * ref_h
                conf = 1 if torch.rand(1, device=ref_img.device) > self.lost_disturb else 0

                ct_int = ct.int()
                # disturb ref_bboxes
                if conf == 0:
                    # ref_centers[idx] = ct
                    ref_bboxes[batch_id][idx, [0, 2]] += ct[0] - ct0[0]
                    ref_bboxes[batch_id][idx, [1, 3]] += ct[1] - ct0[1]
                gen_gaussian_target(ref_hm[batch_id, 0], ct_int, radius, k=conf)

                if torch.rand(1) < self.fp_disturb:
                    ct2 = ct0.clone()
                    ct2[0] = ct2[0] + torch.randn(1, device=ref_img.device) * 0.05 * ref_w
                    ct2[1] = ct2[1] + torch.randn(1, device=ref_img.device) * 0.05 * ref_h
                    ct2_int = ct2.int()
                    gen_gaussian_target(ref_hm[batch_id, 0], ct2_int, radius, k=conf)
            return ref_hm, ref_bboxes

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
            radius = gaussian_radius([torch.ceil(scale_box_h), torch.ceil(scale_box_w)], min_overlap=0.3)
            radius = max(0, int(radius))
            gen_gaussian_target(heatmap[0, 0],
                                ct.int(), radius)
        return heatmap

import torch
from torch import nn

from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from mmdet.models.utils.gaussian_target import get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat, \
    gaussian_radius, gen_gaussian_target


@HEADS.register_module()
class CenterTrackHead(CenterNetHead):
    """Tracking Objects as Points Head. Paper link <https://arxiv.org/abs/2004.01177>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background category. Default: 1.
        use_ltrb (bool): Whether to use ltrb instead of wh to compute bbox from heatmap. Default: True.
        loss_center_heatmap (dict | None): Config of center heatmap loss. Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        loss_tracking (dict | None): Config of tracking loss. Default: L1Loss.
        loss_ltrb_amodal (dict | None): Config of ltrb_amodal loss. Default: L1Loss.
        init_cfg (dict or list[dict], optional): Initialization config dict. Default: None
        train_cfg (dict | None): Training config. Useless in CenterNet, but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes=1,
                 use_ltrb=True,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_tracking=dict(type='L1Loss', loss_weight=1.0),
                 loss_ltrb_amodal=dict(type='L1Loss', loss_weight=0.1),
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):

        super(CenterTrackHead, self).__init__(in_channel,
                                              feat_channel,
                                              num_classes,
                                              loss_center_heatmap,
                                              loss_wh,
                                              loss_offset,
                                              init_cfg=init_cfg,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg, )
        self.use_ltrb = use_ltrb
        self.test_cfg = test_cfg
        self.fp_disturb = train_cfg['fp_disturb']
        self.hm_disturb = train_cfg['hm_disturb']
        self.lost_disturb = train_cfg['lost_disturb']

        self.ltrb_amodal_head = self._build_head(in_channel, feat_channel, 4)
        self.tracking_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_tracking = build_loss(loss_tracking)
        self.loss_ltrb_amodal = build_loss(loss_ltrb_amodal)

    def _affine_transform(self, pts, t):
        """Affine transform

        Args:
            pts (Tensor): points, shape (num_points, 2)
            t: (np.array): map_matrix, shape (2,3)

        Returns: transformed points, shape (num_points, 2)
        """
        new_pts = torch.cat((pts, pts.new_ones(pts.shape[0], 1)), axis=1)
        new_pts = torch.matmul(new_pts, torch.tensor(t, dtype=pts.dtype, device=pts.device).T)
        return new_pts

    def _smooth_sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def forward(self, feat):
        """Forward features.

        Args:
            feat (Tensor): Feature from the upstream network, is a 4D-tensor. Notice
                where is a little difference from CenterNeck. We return a Tensor in
                neck but CenterNeck return a tuple.

        Returns: (dict) : center_heatmap_pred (Tensor), center predict heatmaps,
                            channels number is num_classes.
                          wh_pred (Tensor), wh predicts, the channels number is 2.
                          offset_pred (Tensor): offset predicts, the channels number is 2.
                          ltrb_amodal_pred (Tensor): ltrb amodal predicts, the channels number is 4.
                          tracking_pred (Tensor): tracking predicts, the channels number is 2.
        """
        center_heatmap_pred = self._smooth_sigmoid(self.heatmap_head(feat))
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        ltrb_amodal_pred = self.ltrb_amodal_head(feat)
        tracking_pred = self.tracking_head(feat)
        return dict(
            center_heatmap_pred=center_heatmap_pred,
            wh_pred=wh_pred,
            offset_pred=offset_pred,
            ltrb_amodal_pred=ltrb_amodal_pred,
            tracking_pred=tracking_pred
        )

    def forward_train(self, x, gt_amodal_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_bboxes):
        """

        Args:
            x (Tensor): Feature from neck. Notice where is a little difference from CenterNeck.
                We return a Tensor in neck but CenterNeck return a tuple.
            gt_amodal_bboxes (list[Tensor]): Ground truth bboxes of the image, amodal means the bboxes can be
                outside of the image, len is batch size, shape (num_gts,4)
            gt_labels (list[Tensor]): Ground truth labels of each bbox, len is batch size, shape (num_gts,)
            feat_shape (Torch.Size): feature map size.
            img_shape (Tuple): input img size.
            gt_match_indices: (List[Tensor]): len is batch size, shape (num_gts,)
            ref_bboxes: (list[Tensor]): Ground true bboxes of reference image,
                len is batch size, shape (num_ref_gts, 4)

        Returns:
             (dict): the dict has componets below:
               - loss_center_heatmap
               - loss_wh
               - loss_offset
               - loss_tracking
               - loss_ltrb_amodal
        """
        outs = self(x)
        loss = self.loss(outs, gt_amodal_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_bboxes)
        return loss

    def loss(self, outputs, gt_amodal_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_bboxes):
        """Compute losses of the head.

        Args:
            outputs (dict) : return of self.forward
            gt_amodal_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_shape (Tuple): input img size.
            gt_match_indices: (List[Tensor]): len is batch size, shape (num_gts,)
            gt_match_indices:(List[Tensor]): len is batch size, shape (num_gts,)
            ref_bboxes: (list[Tensor]): Ground true bboxes of reference image,
                len is batch size, shape (num_ref_gts, 4)

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
                - loss_tracking (Tensor): loss of trakcing.
                - loss_ltrb_amodal (Tensor): loss of ltrb amodal.
        """
        targets = self.get_targets(gt_amodal_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_bboxes)
        outputs = outputs
        # predict
        center_heatmap_pred = outputs['center_heatmap_pred']
        wh_pred = outputs['wh_pred']
        offset_pred = outputs['offset_pred']
        ltrb_amodal_pred = outputs['ltrb_amodal_pred']
        tracking_pred = outputs['tracking_pred']
        # target
        center_heatmap_target = targets['center_heatmap_target']
        wh_target = targets['wh_target']
        offset_target = targets['offset_target']
        ltrb_amodal_target = targets['ltrb_amodal_target']
        tracking_target = targets['tracking_target']
        # weight
        wh_offset_target_weight = targets['wh_offset_target_weight']
        ltrb_amodal_target_weight = targets['ltrb_amodal_target_weight']
        tracking_target_weight = targets['tracking_target_weight']
        # avg factor
        avg_factor = targets['avg_factor']
        # loss
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred,
            center_heatmap_target,
            avg_factor=avg_factor)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_tracking = self.loss_wh(
            tracking_pred,
            tracking_target,
            tracking_target_weight,
            avg_factor=avg_factor * 2)
        loss_ltrb_amodal = self.loss_wh(
            ltrb_amodal_pred,
            ltrb_amodal_target,
            ltrb_amodal_target_weight,
            avg_factor=avg_factor * 4)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_tracking=loss_tracking,
            loss_ltrb_amodal=loss_ltrb_amodal)

    def get_targets(self, gt_amodal_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_bboxes):
        """ Compute regression and classification targets in multiple images.
            Compared with Centerneck, there are some new targets.

        Args:
            gt_amodal_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format. Amodal means the bboxes can be
                outside of the image.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape in [h, w, c] format.
            gt_match_indices (list[Tensor]):: bboxes match indices
            ref_bboxes: (list[Tensor]): Ground true bboxes of reference image,
                len is batch size, shape (num_ref_gts, 4)

        Returns:
            (dict): the dict has componets below:
                -  center_heatmap_target (Tensor): Targets of center heatmap, shape (B, num_classes, H, W).
                -  wh_target (Tensor): Targets of wh predict, shape (B, 2, H, W).
                -  wh_offset_target_weight (Tensor): Weights of wh and offset predict, shape (B, 2, H, W).
                -  offset_target (Tensor): Targets of offset predict, shape (B, 2, H, W).
                -  ltrb_amodal_target (Tensor): Targets of ltrb amodal predict, shape (B, 4, H, W).
                -  ltrb_amodal_target_weight (Tensor): Weights of ltrb amodal predict, shape (B, 4, H, W).
                -  tracking_target (Tensor): Targets of tarcking predict, shape (B, 2, H, W).
                -  tracking_target_weight (Tensor): Weights of tracking predict, shape (B, 2, H, W).
                -  avg_factor (float): avarage factor.
        """
        targets = dict()

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_amodal_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        wh_target = gt_amodal_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_amodal_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_amodal_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])

        tracking_target = gt_amodal_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        tracking_target_weight = gt_amodal_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])

        ltrb_amodal_target = gt_amodal_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])
        ltrb_amodal_target_weight = gt_amodal_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])

        for batch_id in range(bs):
            # amodal scale gt bbox
            scale_gt_amodal_bbox = gt_amodal_bboxes[batch_id].clone()
            scale_gt_amodal_bbox[:, [0, 2]] = scale_gt_amodal_bbox[:, [0, 2]] * width_ratio
            scale_gt_amodal_bbox[:, [1, 3]] = scale_gt_amodal_bbox[:, [1, 3]] * height_ratio
            # clipped scale gt bbox
            scale_gt_bbox = gt_amodal_bboxes[batch_id].clone()
            scale_gt_bbox[:, [0, 2]] = torch.clip(scale_gt_bbox[:, [0, 2]] * width_ratio, 0, feat_w - 1)
            scale_gt_bbox[:, [1, 3]] = torch.clip(scale_gt_bbox[:, [1, 3]] * height_ratio, 0, feat_h - 1)
            #  clipped scale ref bbox
            scale_ref_bbox = ref_bboxes[batch_id]
            scale_ref_bbox[:, [0, 2]] = scale_ref_bbox[:, [0, 2]] * width_ratio
            scale_ref_bbox[:, [1, 3]] = scale_ref_bbox[:, [1, 3]] * height_ratio
            # clipped scale gt centers
            scale_gt_centers = (scale_gt_bbox[:, [0, 1]] + scale_gt_bbox[:, [2, 3]]) / 2
            # clipped ref centers
            scale_ref_centers = (scale_ref_bbox[:, [0, 1]] + scale_ref_bbox[:, [2, 3]]) / 2
            # labels
            gt_label = gt_labels[batch_id]

            for j, ct in enumerate(scale_gt_centers):
                ctx, cty = ct
                scale_box_h = scale_gt_bbox[j][3] - scale_gt_bbox[j][1]
                scale_box_w = scale_gt_bbox[j][2] - scale_gt_bbox[j][0]
                if scale_box_h <= 0 or scale_box_w <= 0:
                    continue
                radius = gaussian_radius([torch.ceil(scale_box_h), torch.ceil(scale_box_w)], min_overlap=0.3)
                radius = max(0, int(radius))

                ctx_int, cty_int = ct.int()
                gen_gaussian_target(center_heatmap_target[batch_id, gt_label[j]],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                ltrb_amodal_target[batch_id, 0, cty_int, ctx_int] = scale_gt_amodal_bbox[j, 0] - ctx_int
                ltrb_amodal_target[batch_id, 1, cty_int, ctx_int] = scale_gt_amodal_bbox[j, 1] - cty_int
                ltrb_amodal_target[batch_id, 2, cty_int, ctx_int] = scale_gt_amodal_bbox[j, 2] - ctx_int
                ltrb_amodal_target[batch_id, 3, cty_int, ctx_int] = scale_gt_amodal_bbox[j, 3] - cty_int
                ltrb_amodal_target_weight[batch_id, :, cty_int, ctx_int] = 1

                if gt_match_indices[batch_id][j] != -1:
                    idx = gt_match_indices[batch_id][j]
                    scale_ref_h = scale_ref_bbox[idx][3] - scale_ref_bbox[idx][1]
                    scale_ref_w = scale_ref_bbox[idx][2] - scale_ref_bbox[idx][0]
                    if scale_ref_h <= 0 or scale_ref_w <= 0:
                        continue
                    else:
                        ref_ctx, ref_cty = scale_ref_centers[idx]
                        tracking_target[batch_id, 0, cty_int, ctx_int] = ref_ctx - ctx_int
                        tracking_target[batch_id, 1, cty_int, ctx_int] = ref_cty - cty_int
                        tracking_target_weight[batch_id, :, cty_int, ctx_int] = 1

        targets['center_heatmap_target'] = center_heatmap_target
        targets['wh_target'] = wh_target
        targets['wh_offset_target_weight'] = wh_offset_target_weight
        targets['offset_target'] = offset_target
        targets['ltrb_amodal_target'] = ltrb_amodal_target
        targets['ltrb_amodal_target_weight'] = ltrb_amodal_target_weight
        targets['tracking_target'] = tracking_target
        targets['tracking_target_weight'] = tracking_target_weight
        targets['avg_factor'] = max(1, center_heatmap_target.eq(1).sum())
        return targets

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   tracking_preds,
                   ltrb_amodal_preds,
                   img_metas,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions, center predictions and tracking prediction.
           Compared with Cent
        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            tracking_preds (list[Tensor]): tracking predicts for all levels
                with shape (B, 2, H, W).
            ltrb_amodal_preds (list[Tensor]): ltrb amodal predicts for all levels
                with shape (B, 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            with_nms (bool): If True, do nms before return boxes.
                Default: False. Undo

        Returns:
            list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: The first item is an (n, 5) tensor,
                where 5 represent (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
                The third item is an (n, 5) tensor and its coordinate are input dimensions.
                The fourth item is an (n, 2) tensor represents the center of bboxes.
                The fifth  item is an (n, 2) tensor represents tracking target of bboxes.
        """
        invert_transfrom = [img_meta['invert_transform_affine'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels, batch_centers, tracking_offset = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            tracking_preds[0],
            ltrb_amodal_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)
        batch_det_bboxes_input = batch_det_bboxes.clone()
        bs = batch_det_bboxes.shape[0]
        for batch_id in range(bs):
            batch_det_bboxes[batch_id, :, :2] = self._affine_transform(batch_det_bboxes[batch_id, :, :2],
                                                                       invert_transfrom[batch_id])
            batch_det_bboxes[batch_id, :, 2:-1] = self._affine_transform(batch_det_bboxes[batch_id, :, 2:-1],
                                                                         invert_transfrom[batch_id])
            batch_centers[batch_id, :, :] = self._affine_transform(batch_centers[batch_id, :, :],
                                                                   invert_transfrom[batch_id])
            offset_transform = invert_transfrom[batch_id].copy()
            offset_transform[:, -1] = 0
            tracking_offset[batch_id, :, :] = self._affine_transform(tracking_offset[batch_id, :, :],
                                                                     offset_transform)

        if with_nms:  # todo
            det_results = []
            for (det_bboxes, det_labels, ref_bboxes) in zip(batch_det_bboxes,
                                                            batch_labels, batch_gt_bboxes_with_motion):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                ref_bbox, _ = self._bboxes_nms(ref_bboxes, det_labels,
                                               self.test_cfg)
                det_results.append(tuple([det_bbox, det_label, ref_bbox]))

        else:
            det_results = [
                tuple(bs) for bs in
                zip(batch_det_bboxes, batch_labels, batch_det_bboxes_input, batch_centers, tracking_offset)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       tracking_pred,
                       ltrb_amodal_preds,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): Center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): Wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): Offset predict, shape (B, 2, H, W).
            tracking_pred (Tensor): Tracking predict, shape (B, 2, H, W).
            ltrb_amodal_pred (Tensor): Ltrb amodal predict, shape (B, 2, H, W).
            img_shape (list[int]): Image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterTrackHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
              - batch_centers (Tensor): Centers of bbox, shape (B,k,2).
              - tracking_offset (Tensor): tracking offset, use to predict reference bboxes, shape (B,k,2).
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys0, topk_xs0 = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        offset = transpose_and_gather_feat(offset_pred, batch_index)
        ltrb = transpose_and_gather_feat(ltrb_amodal_preds, batch_index)
        tracking_offset = transpose_and_gather_feat(tracking_pred, batch_index)
        if self.use_ltrb:
            tl_x = (topk_xs0 + ltrb[..., 0]) * (inp_w / width)
            tl_y = (topk_ys0 + ltrb[..., 1]) * (inp_h / height)
            br_x = (topk_xs0 + ltrb[..., 2]) * (inp_w / width)
            br_y = (topk_ys0 + ltrb[..., 3]) * (inp_h / height)
        else:
            wh = transpose_and_gather_feat(wh_pred, batch_index)
            wh[wh < 0] = 0
            topk_xs = topk_xs0 + offset[..., 0]
            topk_ys = topk_ys0 + offset[..., 1]
            tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
            tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
            br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
            br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        batch_centers = torch.stack([topk_xs0 * (inp_w / width), topk_ys0 * (inp_h / height)], dim=2)
        tracking_offset[:, :, 0] *= inp_w / width
        tracking_offset[:, :, 1] *= inp_h / height
        return batch_bboxes, batch_topk_labels, batch_centers, tracking_offset

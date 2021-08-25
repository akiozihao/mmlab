import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.gaussian_target import get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat, \
    gaussian_radius, gen_gaussian_target
from torch import nn


def affine_transform(pt, t):
    new_pt = torch.cat((pt, pt.new_ones(pt.shape[0], 1)), axis=1)
    new_pt = torch.matmul(new_pt, torch.tensor(t, dtype=pt.dtype, device=pt.device).T)
    return new_pt


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _only_neg_loss(pred, gt):
    gt = torch.pow(1 - gt, 4)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
    return neg_loss.sum()


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


@HEADS.register_module()
class CenterTrackHead(BaseModule):
    def __init__(self,
                 heads, head_convs, num_stacks, last_channel, weights,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_tracking=dict(type='L1Loss', loss_weight=1.0),
                 loss_ltrb_amodal=dict(type='L1Loss', loss_weight=0.1),
                 init_cfg=None, train_cfg=None, test_cfg=None):
        super(CenterTrackHead, self).__init__(init_cfg)
        self.use_ltrb = True
        self.test_cfg = test_cfg
        self.fp_disturb = train_cfg['fp_disturb']
        self.hm_disturb = train_cfg['hm_disturb']
        self.lost_disturb = train_cfg['lost_disturb']
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        self.weights = weights
        self.num_classes = 1
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
                out = nn.Conv2d(head_conv[-1], classes,
                                kernel_size=1, stride=1, padding=0, bias=True)
                conv = nn.Conv2d(last_channel, head_conv[0],
                                 kernel_size=head_kernel,
                                 padding=head_kernel // 2, bias=True)
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                           kernel_size=1, bias=True))
                if len(convs) == 1:
                    fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True), out)
                elif len(convs) == 3:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True), out)
                elif len(convs) == 4:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True),
                        convs[3], nn.ReLU(inplace=True), out)
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-4.6)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(last_channel, classes,
                               kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-4.6)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_offset = build_loss(loss_offset)
        self.loss_wh = build_loss(loss_wh)
        self.loss_tracking = build_loss(loss_tracking)
        self.loss_ltrb_amodal = build_loss(loss_ltrb_amodal)

    def forward(self, feats):
        outs = []
        for s in range(self.num_stacks):
            z = {}
            for head in self.heads:
                z[head] = self.__getattr__(head)(feats[s])
            z = self._sigmoid_output(z)
            outs.append(z)
        return outs

    def forward_train(self, x, gt_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_gt_bboxes):
        outs = self(x)
        loss = self.loss(outs, gt_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_gt_bboxes)
        return loss

    # todo ximi
    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def loss(self, outputs, gt_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_gt_bboxes):
        targets = self.get_targets(gt_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_gt_bboxes)
        outputs = outputs[0]

        center_heatmap_pred = outputs['hm']
        offset_pred = outputs['reg']
        wh_pred = outputs['wh']
        tracking_pred = outputs['tracking']
        ltrb_amodal_pred = outputs['ltrb_amodal']

        center_heatmap_target = targets['center_heatmap_target']
        offset_target = targets['offset_target']
        wh_target = targets['wh_target']
        tracking_target = targets['tracking_target']
        ltrb_amodal_target = targets['ltrb_amodal_target']

        wh_offset_target_weight = targets['wh_offset_target_weight']
        ltrb_amodal_target_weight = targets['ltrb_amodal_target_weight']
        tracking_target_weight = targets['tracking_target_weight']

        avg_factor = targets['avg_factor']

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

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape, gt_match_indices, ref_gt_bboxes):
        """
        self, gt_bboxes, gt_labels, feat_shape, img_shape,
                    gt_match_indices,
                    ref_gt_bboxes
        Returns: dict
             img,ref_img,ref_hm
             hm_target,reg_target,tracking_target,wh_target,ltrb_amodal_target,ind
        """
        targets = dict()

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])

        tracking_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        tracking_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])

        ltrb_amodal_target = gt_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])
        ltrb_amodal_target_weight = gt_bboxes[-1].new_zeros([bs, 4, feat_h, feat_w])

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
            # centers
            # clipped scale gt centers
            scale_gt_center_x = (scale_gt_bbox[:, [0]] + scale_gt_bbox[:, [2]]) / 2
            scale_gt_center_y = (scale_gt_bbox[:, [1]] + scale_gt_bbox[:, [3]]) / 2
            # clipped ref centers
            scale_ref_center_x = (scale_ref_bbox[:, [0]] + scale_ref_bbox[:, [2]]) / 2
            scale_ref_center_y = (scale_ref_bbox[:, [1]] + scale_ref_bbox[:, [3]]) / 2
            scale_ref_centers = torch.cat((scale_ref_center_x, scale_ref_center_y), dim=1)
            # labels
            gt_label = gt_labels[batch_id]

            # cat centers
            scale_gt_centers = torch.cat((scale_gt_center_x, scale_gt_center_y), dim=1)

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

        invert_transfrom = [img_meta['invert_transform'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels, batch_gt_bboxes_with_motion = self.decode_heatmap(
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
            batch_det_bboxes[batch_id, :, :2] = affine_transform(batch_det_bboxes[batch_id, :, :2],
                                                                 invert_transfrom[batch_id])
            batch_det_bboxes[batch_id, :, 2:-1] = affine_transform(batch_det_bboxes[batch_id, :, 2:-1],
                                                                   invert_transfrom[batch_id])
            batch_gt_bboxes_with_motion[batch_id, :, :2] = affine_transform(
                batch_gt_bboxes_with_motion[batch_id, :, :2],
                invert_transfrom[batch_id])
            batch_gt_bboxes_with_motion[batch_id, :, 2:-1] = affine_transform(
                batch_gt_bboxes_with_motion[batch_id, :, 2:-1],
                invert_transfrom[batch_id])

        if with_nms:
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
                zip(batch_det_bboxes, batch_labels, batch_gt_bboxes_with_motion, batch_det_bboxes_input)
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
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.
        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
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
        # centers
        topk_xs = topk_xs0 + offset[..., 0]
        topk_ys = topk_ys0 + offset[..., 1]
        tracking_offset = transpose_and_gather_feat(tracking_pred, batch_index)
        with_motion_topk_xs = topk_xs + tracking_offset[..., 0]
        with_motion_topk_ys = topk_ys + tracking_offset[..., 1]
        with_motion_topk_xs0 = topk_xs0 + tracking_offset[..., 0]
        with_motion_topk_ys0 = topk_ys0 + tracking_offset[..., 1]
        if self.use_ltrb:
            tl_x = (topk_xs0 + ltrb[..., 0]) * (inp_w / width)
            tl_y = (topk_ys0 + ltrb[..., 1]) * (inp_h / height)
            br_x = (topk_xs0 + ltrb[..., 2]) * (inp_w / width)
            br_y = (topk_ys0 + ltrb[..., 3]) * (inp_h / height)
            # ref bboxes
            with_motion_tl_x = (with_motion_topk_xs0 + ltrb[..., 0]) * (inp_w / width)
            with_motion_tl_y = (with_motion_topk_ys0 + ltrb[..., 1]) * (inp_h / height)
            with_motion_br_x = (with_motion_topk_xs0 + ltrb[..., 2]) * (inp_w / width)
            with_motion_br_y = (with_motion_topk_ys0 + ltrb[..., 3]) * (inp_h / height)
        else:
            wh = transpose_and_gather_feat(wh_pred, batch_index)
            wh[wh < 0] = 0
            tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
            tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
            br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
            br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
            # ref bboxes
            with_motion_tl_x = (with_motion_topk_xs - wh[..., 0] / 2) * (inp_w / width)
            with_motion_tl_y = (with_motion_topk_ys - wh[..., 1] / 2) * (inp_h / height)
            with_motion_br_x = (with_motion_topk_xs + wh[..., 0] / 2) * (inp_w / width)
            with_motion_br_y = (with_motion_topk_ys + wh[..., 1] / 2) * (inp_h / height)
        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        with_motion_batch_bboxes = torch.stack([with_motion_tl_x, with_motion_tl_y, with_motion_br_x, with_motion_br_y],
                                               dim=2)
        with_motion_batch_bboxes = torch.cat((with_motion_batch_bboxes, batch_scores[..., None]),
                                             dim=-1)
        return batch_bboxes, batch_topk_labels, with_motion_batch_bboxes


class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''

    def __init__(self, opt=None):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        '''
        neg_loss = self.only_neg_loss(out, target)
        pos_pred_pix = _tranpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                   mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

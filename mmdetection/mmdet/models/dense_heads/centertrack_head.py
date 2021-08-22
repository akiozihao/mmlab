from math import sqrt

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models.utils.gaussian_target import get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
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


@HEADS.register_module()
class CenterTrackHead(BaseModule):
    def __init__(self,
                 heads, head_convs, num_stacks, last_channel, weights, init_cfg=None, train_cfg=None, test_cfg=None):
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

    def forward(self, feats):
        outs = []
        for s in range(self.num_stacks):
            z = {}
            for head in self.heads:
                z[head] = self.__getattr__(head)(feats[s])
            z = self._sigmoid_output(z)
            outs.append(z)
        return outs

    def forward_train(self, x, batch):
        outs = self(x)
        loss = self.loss(outs, batch)
        return loss

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def loss(self, outputs, batch):
        losses = {head: 0 for head in self.heads}

        for s in range(self.num_stacks):
            output = outputs[s]

            if 'hm' in output:
                losses['hm'] += self.crit(
                    output['hm'], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / self.num_stacks

            regression_heads = [
                'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                'dep', 'dim', 'amodel_offset', 'velocity']

            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / self.num_stacks

            if 'hm_hp' in output:
                losses['hm_hp'] += self.crit(
                    output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                    batch['hm_hp_mask'], batch['joint']) / self.num_stacks
                if 'hp_offset' in output:
                    losses['hp_offset'] += self.crit_reg(
                        output['hp_offset'], batch['hp_offset_mask'],
                        batch['hp_ind'], batch['hp_offset']) / self.num_stacks

            if 'rot' in output:
                losses['rot'] += self.crit_rot(
                    output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                    batch['rotres']) / self.num_stacks

            if 'nuscenes_att' in output:
                losses['nuscenes_att'] += self.crit_nuscenes_att(
                    output['nuscenes_att'], batch['nuscenes_att_mask'],
                    batch['ind'], batch['nuscenes_att']) / self.num_stacks

        for head in self.heads:
            losses[head] = self.weights[head] * losses[head]

        format_loss = dict()
        for k, v in losses.items():
            nk = 'loss_' + k
            format_loss[nk] = v
        return format_loss
        # losses['tot'] = 0
        # for head in self.heads:
        #     losses['tot'] += self.weights[head] * losses[head]
        #
        # return losses['tot'], losses

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
            kernel=self.test_cfg.local_maximum_kernel,)
        batch_det_bboxes_input = batch_det_bboxes.clone()
        # batch_border = batch_det_bboxes.new_tensor(
        #     border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        # batch_det_bboxes[..., :4] -= batch_border
        # batch_gt_bboxes_with_motion[..., :4] -= batch_border
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

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        offset = transpose_and_gather_feat(offset_pred, batch_index)
        ltrb = transpose_and_gather_feat(ltrb_amodal_preds, batch_index)
        # centers
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tracking_offset = transpose_and_gather_feat(tracking_pred, batch_index)
        with_motion_topk_xs = topk_xs + tracking_offset[..., 0]
        with_motion_topk_ys = topk_ys + tracking_offset[..., 1]
        if self.use_ltrb:
            tl_x = (topk_xs + ltrb[..., 0]) * (inp_w / width)
            tl_y = (topk_ys + ltrb[..., 1]) * (inp_h / height)
            br_x = (topk_xs + ltrb[..., 2]) * (inp_w / width)
            br_y = (topk_ys + ltrb[..., 3]) * (inp_h / height)
            # ref bboxes
            with_motion_tl_x = (with_motion_topk_xs + ltrb[..., 0]) * (inp_w / width)
            with_motion_tl_y = (with_motion_topk_ys + ltrb[..., 1]) * (inp_h / height)
            with_motion_br_x = (with_motion_topk_xs + ltrb[..., 2]) * (inp_w / width)
            with_motion_br_y = (with_motion_topk_ys + ltrb[..., 3]) * (inp_h / height)
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

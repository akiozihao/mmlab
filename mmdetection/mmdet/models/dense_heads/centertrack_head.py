import torch
import torch.nn.functional as F
from mmdet.models import HEADS
from torch import nn


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
class CenterTrackHead(nn.Module):
    def __init__(self,
                 heads, head_convs, num_stacks, last_channel,weights):
        super(CenterTrackHead, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        self.weights=weights
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
        out = []
        for s in range(self.num_stacks):
            z = {}
            for head in self.heads:
                z[head] = self.__getattr__(head)(feats[s])
            out.append(z)
        return out

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def get_target(self):
        # todo
        pass

    def loss(self, outputs, batch):
        losses = {head: 0 for head in self.heads}

        for s in range(self.num_stacks):
            output = outputs[s]
            output = self._sigmoid_output(output)

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

        losses['tot'] = 0
        for head in self.heads:
            losses['tot'] += self.weights[head] * losses[head]

        return losses['tot'], losses


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

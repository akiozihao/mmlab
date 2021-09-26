from collections import OrderedDict
import argparse

import torch
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert origin checkpoint to mm ')
    parser.add_argument('-i', '--input', help='path of origin checkpoint')
    parser.add_argument(
        '-o', '--output', help='path to save mm checkpoint')
    return parser.parse_args()

def transfer_pth(source_pth):
    dst_pth_info = dict()
    dst_state_dict = OrderedDict()
    source = torch.load(source_pth)
    source_state_dict = source['state_dict']
    for k, v in source_state_dict.items():
        type = k.split('.')[0]
        if type == 'base':
            nk, nv = trans_base(k, v)
        elif type == 'dla_up' or type == 'ida_up':
            nk, nv = trans_neck(k, v)
        elif type == 'hm' or type == 'reg' or type == 'wh' \
                or type == 'tracking' or type == 'ltrb_amodal':
            nk, nv = trans_head(k, v)
        else:
            nk, nv = k, v
        nk = 'detector.' + nk
        dst_state_dict[nk] = nv
    for k, v in source.items():
        if k == 'state_dict':
            continue
        else:
            dst_pth_info[k] = v
    dst_pth_info['state_dict'] = dst_state_dict
    return dst_pth_info


def trans_base(k, v):
    l_k = k.split('.')
    l_k[0] = 'backbone'
    if l_k[1] == 'base_layer' or l_k[1] == 'pre_img_layer' or l_k[
            1] == 'pre_hm_layer':
        if l_k[2] == '0':
            l_k[2] = 'conv'
        elif l_k[2] == '1':
            l_k[2] = 'bn'
    if l_k[1] == 'level0' or l_k[1] == 'level1':
        if l_k[2] == '0':
            l_k.insert(-1, 'conv')
        elif l_k[2] == '1':
            l_k[2] = '0'
            l_k.insert(-1, 'bn')
    if l_k[2] == 'root':
        if l_k[3] == 'bn':
            l_k[3] = l_k[3] + '1'
    if l_k[3] == 'root':
        if l_k[4] == 'bn':
            l_k[4] = l_k[4] + '1'
    return '.'.join(str(i) for i in l_k), v


def trans_neck(k, v):
    l_k = k.split('.')
    # if l_k[0] == 'dla_up':
    #     l_k = trans_dla(k)
    # elif l_k[0] == 'ida_up':
    #     l_k = trans_ida(k)
    l_k.insert(0, 'neck')
    return '.'.join(str(i) for i in l_k), v


def trans_ida(k):
    l_k = k.split('.')
    if len(l_k) > 3 and l_k[3] == 'conv_offset_mask':
        l_k[3] = 'conv_offset'
    return l_k


def trans_dla(k):
    l_k = k.split('.')
    if len(l_k) > 4 and l_k[4] == 'conv_offset_mask':
        l_k[4] = 'conv_offset'
    return l_k


def trans_head(k, v):
    l_k = k.split('.')
    if l_k[0] == 'hm':
        l_k[0] = 'heatmap_head'
    elif l_k[0] == 'reg':
        l_k[0] = 'offset_head'
    elif l_k[0] == 'wh':
        l_k[0] = 'wh_head'
    elif l_k[0] == 'tracking':
        l_k[0] = 'tracking_head'
    elif l_k[0] == 'ltrb_amodal':
        l_k[0] = 'ltrb_amodal_head'
    l_k.insert(0, 'bbox_head')
    return '.'.join(str(i) for i in l_k), v


# new_pth_info = transfer_pth(
#     '/home/akio/dev/centertrack_origin/models/mot17_half.pth')
#
# for k, v in new_pth_info['state_dict'].items():
#     print(k, v.shape)
#
# torch.save(new_pth_info, '../models/test_mmlab_old_dcn_old_radius_mot17_half'
#                          '.pth')

if __name__ == '__main__':
    args = parse_args()
    new_pth_info = transfer_pth(args.input)

    torch.save(new_pth_info,args.output)

    print('done')
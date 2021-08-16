from collections import OrderedDict

import torch

neck_st = OrderedDict()


def trans_ida(k):
    l_k = k.split('.')
    if l_k[1] == 'actf':
        l_k[1] = 'bn'
        l_k.pop(-2)
    if len(l_k) > 2 and l_k[2] == 'conv_offset_mask':
        l_k[2] = 'conv_offset'
    return l_k


def trans_dla(k):
    l_k = k.split('.')
    if len(l_k) > 2 and l_k[2] == 'actf':
        l_k[2] = 'bn'
        l_k.pop(-2)
    if len(l_k) > 3 and l_k[3] == 'conv_offset_mask':
        l_k[3] = 'conv_offset'
    return l_k


def trans_neck(source_pth):
    source = torch.load(source_pth)
    source_state_dict = source
    for k, v in source_state_dict.items():
        l_k = k.split('.')
        if l_k[0] == 'dla_up':
            l_k = trans_dla(k)
        elif l_k[0] == 'ida_up':
            l_k = trans_ida(k)

        l_k = '.'.join(str(i) for i in l_k)
        neck_st[l_k] = v
    torch.save(neck_st, 'neck.pt')


trans_neck('/home/akio/Downloads/crowdhuman_split/neck.pt')

# for k, v in neck_st.items():
#     print(k, v.shape)

print('done')

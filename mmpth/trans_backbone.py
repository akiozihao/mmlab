from collections import OrderedDict

import torch

backbone_st = OrderedDict()



def trans_backbone(source_pth):
    source = torch.load(source_pth)
    source_state_dict = source
    for k, v in source_state_dict.items():
        l_k = k.split('.')
        if l_k[0] == 'base_layer' or l_k[0] == 'pre_img_layer' or l_k[0] == 'pre_hm_layer':
            if l_k[1] == '0':
                l_k[1] = 'conv'
            elif l_k[1] == '1':
                l_k[1] = 'bn'
        if l_k[0] == 'level0' or l_k[0] == 'level1':
            if l_k[1] == '0':
                l_k.insert(-1, 'conv')
            elif l_k[1] == '1':
                l_k[1] = '0'
                l_k.insert(-1, 'bn')
        if l_k[1] == 'root':
            if l_k[2] == 'bn':
                l_k[2] = l_k[2] + '1'
        if len(l_k) > 2 and l_k[2] == 'root':
            if len(l_k) > 3 and l_k[3] == 'bn':
                l_k[3] = l_k[3] + '1'
        l_k =  '.'.join(str(i) for i in l_k)
        backbone_st[l_k] = v
    # save
    torch.save(backbone_st,'../models/mmlab_dla34-ba72cf86.pth')



trans_backbone('/home/akio/Downloads/dla34-ba72cf86.pth')

for k,v in backbone_st.items():
    print(k,v.shape)

print('done')


from collections import OrderedDict

import torch

backbone_st = OrderedDict()
neck_st = OrderedDict()
head_st = OrderedDict()


def split_pth(source_pth):
    source = torch.load(source_pth)
    source_state_dict = source['state_dict']
    for k, v in source_state_dict.items():
        k = k.split('.')
        type = k[0]
        nk = '.'.join(k[1:])
        if type == 'backbone':
            backbone_st[nk] = v
        elif type == 'neck':
            neck_st[nk] = v
        elif type == 'bbox_head':
            head_st[nk] = v
        else:
            raise TypeError('type error')

    # save
    # torch.save(backbone_st,'backbone.pt')
    # torch.save(neck_st,'neck.pt')
    # torch.save(head_st,'head.pt')


split_pth('../new_model.pth')

for k,v in backbone_st.items():
    print(k,v.shape)

print('done')


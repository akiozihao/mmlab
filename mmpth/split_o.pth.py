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
        if type == 'base':
            nk = '.'.join(k[1:])
            backbone_st[nk] = v
        elif type in ['dla_up','ida_up']:
            nk = '.'.join(k)
            neck_st[nk] = v
        elif type in ['hm','reg','wh','tracking','ltrb_amodal']:
            nk = '.'.join(k)
            head_st[nk] = v
        else:
            raise TypeError('type error')

    # save
    torch.save(backbone_st,'test_o_backbone.pt')
    torch.save(neck_st,'test_o_neck.pt')
    torch.save(head_st,'test_o_head.pt')


split_pth('/home/akio/dev/centertrack_origin/models/mot17_half.pth')

for k, v in backbone_st.items():
    print(k, v.shape)

print('done')

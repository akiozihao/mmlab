import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CrowdHuman to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def parse_gts(infos):
    outputs = defaultdict(list)
    for info in infos:
        info = json.loads(info)
        for gtbox in info['gtboxes']:
            ins_id = -1
            conf = 1.
            class_id = 1
            visibility = 1.
            iscrowd = 'extra' in gtbox and 'ignore' in gtbox['extra'] and \
                      gtbox['extra']['ignore'] == 1
            ann = dict(
                category_id=1,
                bbox=gtbox['fbox'],
                area=gtbox['fbox'][2] * gtbox['fbox'][3],
                iscrowd=iscrowd,
                visibility=visibility,
                mot_instance_id=ins_id,
                mot_conf=conf,
                mot_class_id=class_id)
            outputs[info['ID']].append(ann)
    return outputs


def main():
    args = parse_args()
    if not osp.exists(args.output):
        os.makedirs(args.output)

    sets = ['train', 'val']
    vid_id, img_id, ann_id, instance_id = 1, 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')
        outputs = defaultdict(list)
        outputs['categories'] = [dict(id=1, name='pedestrian')]
        video_name = ['video_1']
        for video_name in video_name:
            # basic params
            video = dict(
                id=vid_id, name=video_name, fps=-1, width=-1, height=-1)
            # parse annotations
            infos = mmcv.list_from_file(
                f'{args.input}/annotation_{subset}.odgt')
            img2gts = parse_gts(infos)
            img_names = img2gts.keys()
            img_names = sorted(img_names)

            for frame_id, img_name in enumerate(tqdm(img_names)):
                file_name = img_name + '.jpg'
                im = cv2.imread(osp.join(f'{args.input}/Images', file_name))
                im_shape = im.shape
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=file_name,
                    height=im_shape[0],
                    width=im_shape[1],
                    frame_id=frame_id,
                    mot_frame_id=frame_id)
                infos = img2gts[img_name]
                for gt in infos:
                    gt.update(
                        id=ann_id, image_id=img_id, instance_id=instance_id)
                    outputs['annotations'].append(gt)
                    ann_id += 1
                    instance_id += 1
                outputs['images'].append(image)
                img_id += 1
            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = instance_id
        print(f'{subset} has {ins_id} instances.')
        mmcv.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()

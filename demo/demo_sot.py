# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv

from mmtrack.apis import inference_sot, init_model


# python ./demo/demo_sot.py \
#     ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_uav123.py \
#     --input /media/ubuntu/b8a63a15-1ff0-450a-8309-529409c0c254/hgx/sense/video/video10.mp4 \
#     --checkpoint checkpoints/siamese_rpn_r50_20e_uav123_20220420_181845-dc2d4831.pth \
#     --output outputs/video1_siamrpn_uav \
#     --show
# python ./demo/demo_sot.py \
#     ./configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
#     --input /media/ubuntu/b8a63a15-1ff0-450a-8309-529409c0c254/hgx/sense/video/video2.mp4 \
#     --checkpoint checkpoints/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth \
#     --output outputs/video2_siamrpn_lasot \
#     --show
# python ./demo/demo_sot.py \
#     ./configs/sot/stark/stark_st2_r50_50e_lasot.py \
#     --input /media/ubuntu/b8a63a15-1ff0-450a-8309-529409c0c254/hgx/sense/video/video1.mp4 \
#     --checkpoint checkpoints/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth \
#     --output outputs/video1_starkst2_lasot \
#     --show

# init box for each video1~video10:
# [70, 382, 827, 899]
# [1530, 110, 1570, 155]
# [1340, 869, 1590, 973]
# [1334, 98, 1395, 151]
# [1109, 564, 1738, 951]
# [983, 326, 1061, 391]
# [1019, 303, 1088, 360]
# [1215, 147, 1277, 190]
# [224, 541, 270, 579]
# [180, 518, 217, 553]
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    parser.add_argument(
        '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    parser.add_argument('--fps', type=int, help='FPS of the output video')
    parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')
    args = parser.parse_args()

    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    OUT_VIDEO = False
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.input, img)
            img = mmcv.imread(img_path)
        if i == 0:
            if args.gt_bbox_file is not None:
                bboxes = mmcv.list_from_file(args.gt_bbox_file)
                init_bbox = list(map(float, bboxes[0].split(',')))
            else:
                init_bbox = list(cv2.selectROI(args.input, img, False, False))

            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]

        # # [hgx0706] get handcraft box in 1st frame
        print("frame_idx", i, "init_bbox:", init_bbox)      # list of 4 ints in format of xyxy
        # exit()
        result = inference_sot(model, img, init_bbox, frame_id=i)
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img_path.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        model.show_result(
            img,
            result,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            thickness=args.thickness)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(
            f'\nmaking the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    main()

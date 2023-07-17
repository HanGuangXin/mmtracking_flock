# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
from mmcv.ops import batched_nms        # [hgx0713]
from mmdet.core import bbox_overlaps    # [hgx0713]

from mmtrack.apis import inference_mot, init_model, inference_sot
from mmtrack.apis import inference_mot_det, inference_mot_track     # [hgx0711] for mot
from mmtrack.core import outs2results, results2outs     # [hgx0711] for mot
from mmtrack.core.bbox import (bbox_cxcywh_to_x1y1wh, bbox_xyxy_to_x1y1wh,
                               calculate_region_overlap, quad2bbox) # [hgx0711] for sot
from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah  # [hgx0713] for mot
import torch
import numpy as np

# # coco 80 classes demo
# python demo/demo_flock.py \
#     --input /media/ubuntu/b8a63a15-1ff0-450a-8309-529409c0c254/hgx/sense/video/video10.mp4 \
#     --output outputs/video10/flock \
#     --mot_config configs/mot/bytetrack/bytetrack_yolox_x_coco.py \
#     --mot_checkpoint checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
#     --sot_config configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py \
#     --sot_checkpoint checkpoints/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth \
#     --block_length 8 \

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')

    # mot hyper-parameters
    parser.add_argument('--mot_config', help='mot config file')
    parser.add_argument('--mot_checkpoint', help='mot checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')

    # sot hyper-parameters
    parser.add_argument('--sot_config', help='sot Config file')
    parser.add_argument('--sot_checkpoint', help='sot Checkpoint file')
    # parser.add_argument(
    #     '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    # parser.add_argument(
    #     '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    # parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')

    # demo settings
    parser.add_argument(
        '--block_length', type=int, default=8,
        help='The number of frames in each block')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
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
            OUT_VIDEO = False
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

    # load config if needed
    mot_config = mmcv.Config.fromfile(args.mot_config)
    sot_config = mmcv.Config.fromfile(args.sot_config)

    # mot model, including det model and tracker
    mot_model = init_model(args.mot_config, args.mot_checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if i == 134:
            print("111")
        if isinstance(img, str):
            img = osp.join(args.input, img)
        if i % args.block_length == 0:  # [hgx0711] the start frame of the block. det and association
            # # result = inference_mot(mot_model, img, frame_id=i)
            # det
            det_bboxes, det_labels = inference_mot_det(mot_model, img, frame_id=i)
            # init mot tracker
            track_bboxes, track_labels, track_ids, invalid_ids = \
                inference_mot_track(mot_model, img, i, det_bboxes, det_labels)  # [hgx0712] add return invalid_ids

            if i == 0:  # init sot trackers
                n_tracklet = len(track_bboxes)
                sot_models = []
                sot_results = []
                for i_trk in range(n_tracklet):
                    sot_model = init_model(args.sot_config, args.sot_checkpoint, device=args.device)    # init sot model for each tracklet
                    sot_model.id = int(track_ids[i_trk])     # track id
                    sot_model.label = track_labels[i_trk]   # track label
                    init_bbox = [int(track_bboxes[i_trk][0]), int(track_bboxes[i_trk][1]),
                                 int(track_bboxes[i_trk][2]), int(track_bboxes[i_trk][3])]
                    sot_result = inference_sot(sot_model, img, init_bbox, frame_id=i)
                    sot_results.append(sot_result)
                    sot_models.append(sot_model)
            else:   # update the self.memo.bbox in sot tracker
                n_tracklet = len(track_bboxes)
                id2idx = {int(sot_model.id): idx for idx, sot_model in enumerate(sot_models)}
                for i_trk in range(n_tracklet):
                    track_id = int(track_ids[i_trk])
                    track_label = track_labels[i_trk]
                    track_bbox = track_bboxes[i_trk]
                    if track_id in id2idx:  # update existing tracklet
                        sot_models[id2idx[track_id]].memo.bbox = quad2bbox(track_bbox[:-1])
                    else:   # init new tracklet
                        sot_model = init_model(args.sot_config, args.sot_checkpoint, device=args.device)    # init sot model for each tracklet
                        sot_model.id = track_id     # track id
                        sot_model.label = track_label   # track label
                        init_bbox = [int(track_bbox[0]), int(track_bbox[1]),
                                     int(track_bbox[2]), int(track_bbox[3])]
                        sot_result = inference_sot(sot_model, img, init_bbox, frame_id=0)       # frame_id must be 0
                        sot_models.append(sot_model)
            # delete sot trackers w.r.t mot invalid_ids
            if len(invalid_ids) != 0:
                id2idx = {int(sot_model.id): idx for idx, sot_model in enumerate(sot_models)}
                invalid_ids.sort(reverse=True)
                for invalid_id in invalid_ids:
                    if invalid_id in id2idx:
                        sot_models.pop(id2idx[invalid_id])
                    else:
                        continue
            assert set(mot_model.tracker.ids) == set([sot_model.id for sot_model in sot_models]), \
                "tracker of mot and sot should be the same"

            # format results for visualization
            track_results = outs2results(
                bboxes=track_bboxes,
                labels=track_labels,
                ids=track_ids,
                num_classes=mot_model.detector.bbox_head.num_classes)
            det_results = outs2results(
                bboxes=det_bboxes, labels=det_labels, num_classes=mot_model.detector.bbox_head.num_classes)
            mot_result = dict(
                det_bboxes=det_results['bbox_results'],
                track_bboxes=track_results['bbox_results'])
        else:       # run sot in the following frames in the block
            sot_results = []
            det_bboxes = []
            det_labels = []
            for sot_model in sot_models:
                sot_result = inference_sot(sot_model, img, [0, 0, 0, 0], frame_id=i)
                sot_results.append(sot_result)
                det_bboxes.append(sot_result['track_bboxes'])
                det_labels.append(sot_model.label)
            det_bboxes = torch.tensor(np.array(det_bboxes)).to(args.device)
            det_labels = torch.tensor(det_labels).to(args.device)

            # nms for tracklets drift of sot model before association
            dets, keep = batched_nms(det_bboxes[:, :-1], det_bboxes[:, -1], det_labels,
                                     mot_config.model.detector.test_cfg.nms, class_agnostic=True)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

            # association
            track_bboxes, track_labels, track_ids, invalid_ids = inference_mot_track(mot_model, img, i, det_bboxes, det_labels)

            # delete sot trackers w.r.t mot invalid_ids
            if len(invalid_ids) != 0:
                id2idx = {int(sot_model.id): idx for idx, sot_model in enumerate(sot_models)}
                invalid_ids.sort(reverse=True)
                for invalid_id in invalid_ids:
                    if invalid_id in id2idx:
                        sot_models.pop(id2idx[invalid_id])
                    else:
                        continue
            # do not generate new tracklets for when using sot as det
            for invalid_id in list(set(mot_model.tracker.ids) - set([sot_model.id for sot_model in sot_models])):
                mot_model.tracker.tracks.pop(invalid_id)
            assert set(mot_model.tracker.ids) == set([sot_model.id for sot_model in sot_models]), \
                "tracker of mot {} and sot {} should be the same.".format\
                    (set(mot_model.tracker.ids), set([sot_model.id for sot_model in sot_models]))
            # mot {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
            # sot {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}

            # update search location with kalman filter estimation
            mot_ids = mot_model.tracker.ids
            id2idx = {int(sot_model.id): idx for idx, sot_model in enumerate(sot_models)}
            for mot_id in mot_ids:
                # track_id = int(track_ids[i_trk])
                # track_label = track_labels[i_trk]
                kf_bbox = bbox_cxcyah_to_xyxy(torch.from_numpy(mot_model.tracker.tracks[mot_id].mean[:4][None]).to(det_bboxes))
                sot_models[id2idx[mot_id]].memo.bbox = quad2bbox(kf_bbox)

            # format results for visualization
            track_results = outs2results(
                bboxes=track_bboxes,
                labels=track_labels,
                ids=track_ids,
                num_classes=mot_model.detector.bbox_head.num_classes)
            det_results = outs2results(
                bboxes=det_bboxes, labels=det_labels, num_classes=mot_model.detector.bbox_head.num_classes)
            mot_result = dict(
                det_bboxes=det_results['bbox_results'],
                track_bboxes=track_results['bbox_results'])
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        mot_model.show_result(
            img,
            mot_result,
            score_thr=args.score_thr,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    main()

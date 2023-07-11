# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_mot, inference_sot, inference_vid, init_model
from .test import multi_gpu_test, single_gpu_test
from .train import init_random_seed, train_model
from .inference import inference_mot_det, inference_mot_track   # [hgx0711] decouple det and track

__all__ = [
    'init_model', 'multi_gpu_test', 'single_gpu_test', 'train_model',
    'inference_mot', 'inference_sot', 'inference_vid', 'init_random_seed',
    'inference_mot_det', 'inference_mot_track'      # [hgx0711] decouple det and track
]

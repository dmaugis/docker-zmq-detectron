#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

import zmq
import json
import numpy as np
import zmqnparray as zmqa


from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils



c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='results',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: png)',
        default='png',
        type=str
    )
    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()


import pycocotools.mask as mask_util
import numpy as np

def do_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9,reply=None):

    masks=None
    classes=None

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        reply["status"] = 0
        reply["boxes"] = None
        #reply["keypoints"] = None
        reply["classes"] = None
        return None

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)



    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    # Empty image
    s0, s1, _ = im.shape
    result = np.empty(shape=(s0, s1), dtype=np.uint8)

    _boxes=[]
    _classes=[]
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        # score too low => skip
        if score < thresh:
            continue
        # only humans
        #if classes[i] != 1:
        #    continue
        if segms is not None and len(segms) > i:
            result=np.maximum(result,i*masks[..., i])
            _boxes.append(boxes[i].tolist())
            _classes.append(classes[i])

    reply["status"]=0
    reply["boxes"]=_boxes
    #reply["keypoints"]=keypoints.tolist()
    reply["classes"]=_classes

    return result



def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    logger.info(
        ' \ Note: inference on the first image will be slower than the '
        'rest (caches and auto-tuning need to warm up)'
    )

    while True:
        reply = {}
        #  Wait for next request from client
        im,extra = zmqa.recv(socket)
        if extra is not None and 'fname' in extra:
            print("Received request %s" % extra)
            reply['fname']=extra['fname']
        else:
            print("Received request %s" % extra)
            reply['fname'] = 'input.jpg'
        # set file name
        im_name=reply['fname']
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        masks=do_one_image_opencv(
            im,  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            thresh=0.7,
            reply=reply
        )
        zmqa.send(socket, masks,extra=reply)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)

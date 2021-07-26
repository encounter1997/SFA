# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .coco import build_cityscapes_cocostyle, build_sim10k_cocostyle
from .coco import build_foggycity_cocostyle, build_citycaronly_cocostyle, build_bdd100k_cocostyle
from .coco import build_city2foggy_cocostyle,  build_sim2city_cocostyle,  build_city2bdd_cocostyle


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    # source only
    if args.dataset_file == 'cityscapes':
        return build_cityscapes_cocostyle(image_set, args)
    if args.dataset_file == 'sim10k':
        return build_sim10k_cocostyle(image_set, args)

    # oracle
    if args.dataset_file == 'foggycity':
        return build_foggycity_cocostyle(image_set, args)
    if args.dataset_file == 'city_caronly':
        return build_citycaronly_cocostyle(image_set, args)
    if args.dataset_file == 'bdd100k':
        return build_bdd100k_cocostyle(image_set, args)

    # da
    if args.dataset_file == 'city2foggy':
        return build_city2foggy_cocostyle(image_set, args)
    if args.dataset_file == 'sim2city':
        return build_sim2city_cocostyle(image_set, args)
    if args.dataset_file == 'city2bdd':
        return build_city2bdd_cocostyle(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

## This file refer to The PR of vedanuj https://github.com/facebookresearch/grid-feats-vqa/pull/3/

"""
Region features extraction script.
"""
import argparse
import os
import torch
import tqdm
import time
import pickle
import logging
import datetime
from fvcore.common.file_io import PathManager

import numpy as np
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup, default_argument_parser
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
    build_detection_test_loader_for_images
)
from grid_feats.evaluator import CustomEvaluator

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper["coco_2014_train"] = "train2014"
dataset_to_folder_mapper["coco_2014_val"] = "val2014"
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper["coco_2015_test"] = "test2015"

def inference_on_dataset(cfg, model, data_loader, dataset_name, output_folder):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    total = len(data_loader)  # inference data loader must have a fixed length
    logger.info("Start inference on {} images".format(total))

    evaluator = CustomEvaluator(dataset_name, cfg, True, output_folder)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def inference(cfg, model, dataset_name, dataset_path):
    with inference_context(model):
        if dataset_name not in dataset_to_folder_mapper:
            dataset_to_folder_mapper[dataset_name] = dataset_name
            set_metadata(dataset_name)
            data_loader = build_detection_test_loader_for_images(cfg, dataset_path)
        else:
            data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        
        dump_folder = os.path.join(
            cfg.OUTPUT_DIR, "my_inference", dataset_to_folder_mapper[dataset_name]
        )
        PathManager.mkdirs(dump_folder)
        
        

        inference_on_dataset(cfg, model, data_loader, dataset_name, dump_folder)

def set_metadata(dataset_name):
    metadata = MetadataCatalog.get(dataset_name)

    with open('vg_thing_attrs.pkl', 'rb') as f:
        thing_attrs = pickle.load(f)
    thing_attrs_id_to_contiguous_id = {i: i for i, _ in enumerate(thing_attrs)}
    with open('vg_thing_classes.pkl', 'rb') as f:
        thing_classes = pickle.load(f)
    thing_dataset_id_to_contiguous_id = {i: i for i, _ in enumerate(thing_classes)}
    metadata.set(
        evaluator_type="custom",
        thing_attrs=thing_attrs,
        thing_attrs_id_to_contiguous_id=thing_attrs_id_to_contiguous_id,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id
    )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    inference(cfg, model, args.dataset, args.dataset_path)

def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Region feature extraction")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="coco_2014_train",
        choices=["coco_2014_train", "coco_2014_val", "coco_2015_test"],
    )
    parser.add_argument(
        "--dataset-path",
        help="path to image folder dataset, if not provided expects a detection dataset",
        default="",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    """
    python extract_region_feature.py --config-file <config> --dataset <dataset> --dataset-path <path_to_dataset_images_dir>
    """
    parser = default_argument_parser()
    
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="coco_2014_train",
        # choices=["coco_2014_train", "coco_2014_val", "coco_2015_test", "VIST", "VQG"],
    )
    parser.add_argument(
        "--dataset-path",
        help="path to image folder dataset, if not provided expects a detection dataset",
        default="",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()   
    print("Command Line Args:", args)
    main(args)
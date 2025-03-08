#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as BaseExp

from yolox.data import get_yolox_datadir

__all__ = ["Exp", "check_exp_value"]


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 17
        # factor of model depth
        self.depth = 1.33
        # factor of model width
        self.width = 1.25
        
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.input_size = (480, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = os.path.join(get_yolox_datadir(), "17flowers-voc", "voc")

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15

        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 50
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (480, 640)

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import FlowersDetection, TrainTransform

        return FlowersDetection(
            data_dir=self.data_dir,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import FlowersDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return FlowersDetection(
            data_dir=self.data_dir,
            split="validation",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )


def check_exp_value(exp: Exp):
    h, w = exp.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"

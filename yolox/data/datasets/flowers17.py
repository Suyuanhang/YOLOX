#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import glob
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import CacheDataset, cache_read_img
from yolox.data.datasets.flowers17_classes import FLOWERS17_CLASSES
from yolox.data.datasets.voc import AnnotationTransform


class FlowersDection(CacheDataset):

    """
    Flowers17 Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to the dataset folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'Flowers17')
    """

    def __init__(
        self,
        data_dir,
        data_split_file="datasplits.mat",
        img_size=(416, 416),
        preproc=None,
        target_transform=AnnotationTransform(),
        dataset_name="Flowers17",
        cache=False,
        cache_type="ram",
    ):
        self.root = data_dir
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(self.root, "%s.xml")
        self._imgpath = os.path.join(self.root, "%s.jpg")
        self._classes = FLOWERS17_CLASSES
        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(FLOWERS17_CLASSES)
        ]
        self.class_ids = list(range(len(FLOWERS17_CLASSES)))
        self.data_splits {"train": [], "validation": [], "test": []}
        data_split_file_full_path = os.path.join(self.root, data_split_file)
        all_annotated_basenames = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(self.root, "*.xml"))]
        if os.path.exists(data_split_file_full_path):
            import scipy.io
            index_num_of_digits = len(all_annotated_basenames.split('_', 1)[1])
            splits = scipy.io.loadmat(data_split_file_full_path)
            # use both train and val for training
            self.data_splits["train"].extend(splits['trn1'].tolist()[0])
            self.data_splits["train"].extend(splits['val1'].tolist()[0])
            self.data_splits["train"] = ['image_'+str(ind).zfill(index_num_of_digits) for ind in self.data_splits["train"]]
            self.data_splits["train"] = list(set(all_annotated_basenames) & set(self.data_splits["train"]))
            # use test set for validation
            self.data_splits["validation"].extend(splits['tst1'].tolist()[0])
            self.data_splits["validation"] = ['image_'+str(ind).zfill(index_num_of_digits) for ind in self.data_splits["validation"]]
            self.data_splits["validation"] = list(set(all_annotated_basenames) & set(self.data_splits["validation"]))
        else:
            raise ValueError(f"{data_split_file} is not found.")
        
        self.ids = self.data_splits["train"]
        self.num_imgs = len(self.ids)
        self.annotations = self._load_coco_annotations()

        path_filename = [
            (self._imgpath % self.ids[i]).split(self.root + "/")[1]
            for i in range(self.num_imgs)
        ]

        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_{self.name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )


    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(self.num_imgs)]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {self._imgpath % img_id} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)

        return img, target, img_info, index

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_flowers_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_flowers_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "Flowers17")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_flowers_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(FLOWERS17_CLASSES):
            if cls == "__background__":
                continue
            print("Writing {} FLOWERS17 results file".format(cls))
            filename = self._get_flowers_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = self.root
        annopath = os.path.join(rootpath, "{:s}.xml")
        cachedir = os.path.join(
            self.root, "annotations_cache", "Flowers17"
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(FLOWERS17_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_flowers_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                self.data_splits["validation"],
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)

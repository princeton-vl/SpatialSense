import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import random
import numpy as np

np.random.seed(0)
from util import phrase2vec, onehot
import os.path
import cv2
import math
import sys
import pickle
from PIL import Image, ImageDraw
from util import read_img
from collections import defaultdict
import pdb


class SpatialDataset(Dataset):
    def __init__(self, split, load_image, args):
        super().__init__()
        self.split = split
        self.load_image = load_image
        self.args = args

        self.annotations = []
        for img in json.load(open(args.datapath)):
            if img["split"] in split.split("_"):
                for annot in img["annotations"]:
                    annot["url"] = img["url"]
                    annot["height"] = img["height"]
                    annot["width"] = img["width"]
                    annot["subject"]["bbox"] = self.fix_bbox(
                        annot["subject"]["bbox"], img["height"], img["width"]
                    )
                    annot["object"]["bbox"] = self.fix_bbox(
                        annot["object"]["bbox"], img["height"], img["width"]
                    )
                    self.annotations.append(annot)

        print("%d relations in %s" % (len(self.annotations), split))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        args = self.args
        annot = self.annotations[idx]

        t_s = self._getT(annot["subject"]["bbox"], annot["object"]["bbox"])
        t_o = self._getT(annot["object"]["bbox"], annot["subject"]["bbox"])

        xs0, xs1, ys0, ys1 = annot["subject"]["bbox"]
        xo0, xo1, yo0, yo1 = annot["object"]["bbox"]

        datum = {
            "url": annot["url"],
            "_id": annot["_id"],
            "subject": {
                "name": annot["subject"]["name"],
                "embedding": phrase2vec(
                    annot["subject"]["name"], self.args.max_phrase_len, 300
                ),
                "bbox": np.asarray(
                    [
                        xs0 / annot["height"],
                        xs1 / annot["height"],
                        ys0 / annot["width"],
                        ys1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_s, dtype=np.float32),
            },
            "object": {
                "name": annot["object"]["name"],
                "embedding": phrase2vec(
                    annot["object"]["name"], self.args.max_phrase_len, 300
                ),
                "bbox": np.asarray(
                    [
                        xo0 / annot["height"],
                        xo1 / annot["height"],
                        yo0 / annot["width"],
                        yo1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_o, dtype=np.float32),
            },
            "label": np.asarray([[annot["label"]]], dtype=np.float32),
            "predicate": onehot(args.predicate_categories.index(annot["predicate"]), 9),
            "predicate_name": annot["predicate"],
        }

        if self.split == "test":
            del datum["label"]

        if self.load_image:
            img = read_img(annot["url"], self.args.imagepath)
            ih, iw = img.shape[:2]

            if "train" in self.split:
                t_bbox = transforms.Compose(
                    [
                        transforms.ToPILImage("RGB"),
                        transforms.Pad(4, padding_mode="edge"),
                        transforms.RandomResizedCrop(32, scale=(0.75, 0.85)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                t_bbox = transforms.Compose(
                    [
                        transforms.ToPILImage("RGB"),
                        transforms.Pad(4, padding_mode="edge"),
                        transforms.CenterCrop(32),
                        transforms.ToTensor(),
                    ]
                )
            bbox_mask = np.stack(
                [
                    self._getDualMask(ih, iw, annot["subject"]["bbox"], 32).astype(
                        np.uint8
                    ),
                    self._getDualMask(ih, iw, annot["object"]["bbox"], 32).astype(
                        np.uint8
                    ),
                    np.zeros((32, 32), dtype=np.uint8),
                ],
                2,
            )
            bbox_mask = t_bbox(bbox_mask)[:2].float() / 255.0
            datum["bbox_mask"] = bbox_mask

            union_bbox = self.enlarge(
                self._getUnionBBox(
                    annot["subject"]["bbox"], annot["object"]["bbox"], ih, iw
                ),
                1.25,
                ih,
                iw,
            )
            if "train" in self.split:
                t_bboximg = transforms.Compose(
                    [
                        transforms.ToPILImage("RGB"),
                        transforms.RandomResizedCrop(224, scale=(0.75, 0.85)),
                        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                        transforms.ToTensor(),
                    ]
                )
            else:
                t_bboximg = transforms.Compose(
                    [
                        transforms.ToPILImage("RGB"),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ]
                )
            bbox_img = t_bboximg(self._getAppr(img, union_bbox))
            datum["bbox_img"] = bbox_img

            if "train" in self.split:
                t_fullimg = transforms.Compose(
                    [
                        transforms.ToPILImage("RGB"),
                        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                        transforms.ToTensor(),
                    ]
                )
            else:
                t_fullimg = transforms.Compose(
                    [transforms.ToPILImage("RGB"), transforms.ToTensor(),]
                )
            if self.args.model == "vipcnn":
                img_size = 400
            elif self.args.model == "pprfcn":
                img_size = 720
            else:
                img_size = 224
            datum["full_img"] = t_fullimg(self._getAppr(img, [0, ih, 0, iw], img_size))

        return datum

    def enlarge(self, bbox, factor, ih, iw):
        height = bbox[1] - bbox[0]
        width = bbox[3] - bbox[2]
        assert height > 0 and width > 0
        return [
            max(0, int(bbox[0] - (factor - 1.0) * height / 2.0)),
            min(ih, int(bbox[1] + (factor - 1.0) * height / 2.0)),
            max(0, int(bbox[2] - (factor - 1.0) * width / 2.0)),
            min(iw, int(bbox[3] + (factor - 1.0) * width / 2.0)),
        ]

    def _getAppr(self, im, bb, out_size=224.0):
        subim = im[bb[0] : bb[1], bb[2] : bb[3], :]
        subim = cv2.resize(
            subim,
            None,
            None,
            out_size / subim.shape[1],
            out_size / subim.shape[0],
            interpolation=cv2.INTER_LINEAR,
        )
        subim = (subim / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return subim.astype(np.float32, copy=False)

    def _getUnionBBox(self, aBB, bBB, ih, iw, margin=10):
        return [
            max(0, min(aBB[0], bBB[0]) - margin),
            min(ih, max(aBB[1], bBB[1]) + margin),
            max(0, min(aBB[2], bBB[2]) - margin),
            min(iw, max(aBB[3], bBB[3]) + margin),
        ]

    def _getDualMask(self, ih, iw, bb, heatmap_size=32):
        rh = float(heatmap_size) / ih
        rw = float(heatmap_size) / iw
        x1 = max(0, int(math.floor(bb[0] * rh)))
        x2 = min(heatmap_size, int(math.ceil(bb[1] * rh)))
        y1 = max(0, int(math.floor(bb[2] * rw)))
        y2 = min(heatmap_size, int(math.ceil(bb[3] * rw)))
        mask = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
        mask[x1:x2, y1:y2] = 255
        # assert(mask.sum() == (y2 - y1) * (x2 - x1))
        return mask

    def _getT(self, bbox1, bbox2):
        h1 = bbox1[1] - bbox1[0]
        w1 = bbox1[3] - bbox1[2]
        h2 = bbox2[1] - bbox2[0]
        w2 = bbox2[3] - bbox2[2]
        return [
            (bbox1[0] - bbox2[0]) / float(h2),
            (bbox1[2] - bbox2[2]) / float(w2),
            math.log(h1 / float(h2)),
            math.log(w1 / float(w2)),
        ]

    def fix_bbox(self, bbox, ih, iw):
        if bbox[1] - bbox[0] < 20:
            if bbox[0] > 10:
                bbox[0] -= 10
            if bbox[1] < ih - 10:
                bbox[1] += 10

        if bbox[3] - bbox[2] < 20:
            if bbox[2] > 10:
                bbox[2] -= 10
            if bbox[3] < iw - 10:
                bbox[3] += 1
        return bbox


def create_dataloader(split, load_image, args):
    dataset = SpatialDataset(split, load_image, args)
    return DataLoader(
        dataset,
        args.batchsize,
        num_workers=args.num_workers,
        shuffle=True if split.startswith("train") else False,
        pin_memory=torch.cuda.is_available(),
    )

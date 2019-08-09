import math
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .vipcnn import union_bbox, intersection_bbox

import pdb


def subj_obj_pool(prs_maps, bbox):
    batchsize = prs_maps.size(0)
    feature = torch.Tensor(batchsize, 9, 3, 3).to(prs_maps.device)
    for n in range(batchsize):
        cell_height = (bbox[n, 1] - bbox[n, 0]) / 3.
        cell_width = (bbox[n, 3] - bbox[n, 2]) / 3.
        for i in range(3):
            for j in range(3):
                bbox_cell = [int(prs_maps.size(2) * (bbox[n, 0] + i * cell_height).item()), 
                             int(prs_maps.size(2) * (bbox[n, 0] + (i + 1) * cell_height).item()),
                             int(prs_maps.size(3) * (bbox[n, 2] + j * cell_width).item()), 
                             int(prs_maps.size(3) * (bbox[n, 2] + (j + 1) * cell_width).item())]
                if bbox_cell[1] <= bbox_cell[0]:
                    if bbox_cell[1] < prs_maps.size(2) - 1:
                        bbox_cell[1] += 1
                    else:
                        bbox_cell[0] -= 1
                if bbox_cell[3] <= bbox_cell[2]:
                    if bbox_cell[3] < prs_maps.size(3):
                        bbox_cell[3] += 1
                    else:
                        bbox_cell[2] -= 1
                start = i * 27 + j * 9
                cell_input = prs_maps[n, start : start + 9, bbox_cell[0] : bbox_cell[1], bbox_cell[2] : bbox_cell[3]]
                feature[n, :, i, j] = F.avg_pool2d(cell_input, (cell_input.size(1), cell_input.size(2))).squeeze()
    return feature


def joint_pool(prs_maps_joint_s, prs_maps_joint_o, bbox_s, bbox_o):
    bbox_union = union_bbox(bbox_s, bbox_o)
    batchsize = prs_maps_joint_s.size(0)
    feature = torch.zeros(batchsize, 9, 3, 3).to(prs_maps_joint_s.device)
    for n in range(batchsize):
        cell_height = (bbox_union[n, 1] - bbox_union[n, 0]) / 3.
        cell_width = (bbox_union[n, 3] - bbox_union[n, 2]) / 3.
        for i in range(3):
            for j in range(3):
                start = i * 27 + j * 9
                bbox_cell = [(bbox_union[n, 0] + i * cell_height).item(), (bbox_union[n, 0] + (i + 1) * cell_height).item(),
                             (bbox_union[n, 2] + j * cell_width).item(), (bbox_union[n, 2] + (j + 1) * cell_width).item()]
                # subject
                I_s = intersection_bbox(bbox_s[n].unsqueeze(0), torch.Tensor(bbox_cell).to(bbox_s.device).unsqueeze(0)).squeeze()
                I_s = [int(prs_maps_joint_s.size(2) * I_s[0].item()), int(prs_maps_joint_s.size(2) * I_s[1].item()), 
                       int(prs_maps_joint_s.size(3) * I_s[2].item()), int(prs_maps_joint_s.size(3) * I_s[3].item())]
                if I_s[0] < I_s[1] and I_s[2] < I_s[3]:
                    cell_input = prs_maps_joint_s[n, start : start + 9, I_s[0] : I_s[1], I_s[2] : I_s[3]]
                    feature[n, :, i, j] += F.avg_pool2d(cell_input, (cell_input.size(1), cell_input.size(2))).squeeze()
                # object
                I_o = intersection_bbox(bbox_o[n].unsqueeze(0), torch.Tensor(bbox_cell).to(bbox_o.device).unsqueeze(0)).squeeze()
                I_o = [int(prs_maps_joint_o.size(2) * I_o[0].item()), int(prs_maps_joint_o.size(2) * I_o[1].item()),
                       int(prs_maps_joint_o.size(3) * I_o[2].item()), int(prs_maps_joint_o.size(3) * I_o[3].item())]
                if I_o[0] < I_o[1] and I_o[2] < I_o[3]:
                    cell_input = prs_maps_joint_o[n, start : start + 9, I_o[0] : I_o[1], I_o[2] : I_o[3]]
                    feature[n, :, i, j] += F.avg_pool2d(cell_input, (cell_input.size(1), cell_input.size(2))).squeeze()
 
    return feature


class PPRFCN(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        if backbone == 'resnet18':
            self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2],
                                        nn.Conv2d(512, 256, kernel_size=1, stride=1))
        elif backbone == 'resnet101':
            self.resnet = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-2],
                                        nn.Conv2d(2048, 256, kernel_size=1, stride=1))

        self.conv = nn.Conv2d(256, 324, kernel_size=1, stride=1)  # subject : object : joint_s : joint_o 4 k^2 R
        self.batchnorm = nn.BatchNorm2d(324)
        self.fc = nn.Sequential(nn.Linear(81, 40),
                                nn.BatchNorm1d(40),
                                nn.Linear(40, 9))


    def forward(self, img, bbox_s, bbox_o, predicate):
        shared_feature_maps = self.resnet(img)  # batchsize x 256 x 23 x 23
        prs_maps = self.batchnorm(self.conv(shared_feature_maps))  # batchsize x 324 x 23 x 23
        prs_maps_s = prs_maps[:, :81, :, :] 
        prs_maps_o = prs_maps[:, 81:162, :, :]
        prs_maps_joint_s = prs_maps[:, 162:243, :, :]
        prs_maps_joint_o =  prs_maps[:, 243:, :, :]

        # subject/object pooling
        feature_s = subj_obj_pool(prs_maps_s, bbox_s)  # 3 x 3 x R
        feature_o = subj_obj_pool(prs_maps_o, bbox_o)  # 3 x 3 x R

        # joint pooling
        feature_joint = joint_pool(prs_maps_joint_s, prs_maps_joint_o, bbox_s, bbox_o)

        # vote
        #logits = (F.avg_pool2d(feature_s, 3) + F.avg_pool2d(feature_o, 3) + F.avg_pool2d(feature_joint, 3)).squeeze()
        #
        logits = self.fc((feature_s + feature_o + feature_joint).view(feature_s.size(0), -1))

        return torch.sum(logits * predicate, 1)

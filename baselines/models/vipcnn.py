import math
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pdb

def roi_pool(x, rois, out_size):
    out = torch.Tensor(x.size(0), x.size(1), out_size, out_size).to(x.device)
    for i in range(x.size(0)):
        bbox = [int(rois[i][0] * x.size(2)), int(rois[i][1] * x.size(2)), int(rois[i][2] * x.size(3)), int(rois[i][3] * x.size(3))]
        if bbox[1] <= bbox[0]:
            if bbox[0] > 0:
                bbox[0] -= 1
            else:
                bbox[1] += 1
        if bbox[3] <= bbox[2]:
            if bbox[2] > 0:
                bbox[2] -= 1
            else:
                bbox[3] += 1
        l = out_size * math.ceil(max(bbox[2], bbox[3]) / out_size)
        out[i] = F.max_pool2d(F.interpolate(x[i:(i + 1), :, bbox[0]:bbox[1], bbox[2]:bbox[3]], size=l), kernel_size=int(l / out_size))
    return out


def union_bbox(bbox_a, bbox_b):
    return torch.cat([torch.min(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1), 
                      torch.max(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1), 
                      torch.min(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1), 
                      torch.max(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1)], dim=1)


def intersection_bbox(bbox_a, bbox_b):
    return torch.cat([torch.max(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
                      torch.min(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
                      torch.max(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
                      torch.min(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1)], dim=1)


class VipCNN(nn.Module):

    def __init__(self, roi_size, backbone):
        super().__init__()
        self.roi_size = roi_size

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.pmps1_conv_so = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
            self.pmps1_conv_p = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.pmps1_conv_so = nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1)
            self.pmps1_conv_p = nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1)

        children = list(resnet.children())
        self.shared_conv_layers = nn.Sequential(*children[:7])

        self.pre_pmps1_so = children[7]
        self.pre_pmps1_p = copy.deepcopy(self.pre_pmps1_so)
       
        self.pmps1_gather_batchnorm_so = nn.BatchNorm2d(128)
        self.pmps1_gather_batchnorm_p = nn.BatchNorm2d(128)
        self.pmps1_conv_so2p = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pmps1_conv_p2s = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) 
        self.pmps1_conv_p2o = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pmps1_broadcast_batchnorm_p = nn.BatchNorm2d(128)
        self.pmps1_broadcast_batchnorm_s = nn.BatchNorm2d(128)
        self.pmps1_broadcast_batchnorm_o = nn.BatchNorm2d(128)

        self.pmps2_gather_linear_so = nn.Linear(256 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_gather_linear_p = nn.Linear(256 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_linear_s2p = nn.Linear(32 * roi_size * roi_size, 32 * roi_size * roi_size)
        self.pmps2_linear_o2p = nn.Linear(32 * roi_size * roi_size, 32 * roi_size * roi_size)
        #self.pmps2_broadcast_linear_so = nn.Linear(64 * roi_size * roi_size, 8 * roi_size * roi_size)
        self.pmps2_broadcast_linear_p = nn.Linear(32 * roi_size * roi_size, 4 * roi_size * roi_size)
        self.pmps2_gather_batchnorm_s = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_gather_batchnorm_o = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_gather_batchnorm_p = nn.BatchNorm1d(32 * roi_size * roi_size)
        self.pmps2_broadcast_batchnorm_p = nn.BatchNorm1d(4 * roi_size * roi_size)

        self.fc = nn.Linear(4 * roi_size * roi_size, 9)


    def forward(self, img, bbox_s, bbox_o, predicate):
        shared_feature_maps = self.shared_conv_layers(img)
        
        pre_pmps1_feature_so = self.pre_pmps1_so(shared_feature_maps)
        pre_pmps1_feature_p = self.pre_pmps1_p(shared_feature_maps)

        # gather
        pmps1_feature_so = self.pmps1_conv_so(pre_pmps1_feature_so)
        pmps1_feature_p = self.pmps1_conv_p(pre_pmps1_feature_p)
        pmps1_gather_so = F.relu(self.pmps1_gather_batchnorm_so(pmps1_feature_so))
        pmps1_gather_p = F.relu(self.pmps1_gather_batchnorm_p(pmps1_feature_p + self.pmps1_conv_so2p(pmps1_gather_so)))

        # braodcast
        pmps1_broadcast_p = F.relu(self.pmps1_broadcast_batchnorm_p(pmps1_feature_p))
        pmps1_broadcast_s = F.relu(self.pmps1_broadcast_batchnorm_s(pmps1_feature_so + self.pmps1_conv_p2s(pmps1_broadcast_p)))
        pmps1_broadcast_o = F.relu(self.pmps1_broadcast_batchnorm_o(pmps1_feature_so + self.pmps1_conv_p2o(pmps1_broadcast_p)))
        
        # concat
        post_pmps1_feature_s = torch.cat([pmps1_gather_so, pmps1_broadcast_s], dim=1)
        post_pmps1_feature_o = torch.cat([pmps1_gather_so, pmps1_broadcast_o], dim=1)
        post_pmps1_feature_p = torch.cat([pmps1_gather_p, pmps1_broadcast_p], dim=1)

        # RoI pooling
        post_pool_feature_s = roi_pool(post_pmps1_feature_s, bbox_s, self.roi_size)
        post_pool_feature_s = post_pool_feature_s.view(post_pool_feature_s.size(0), -1)
        post_pool_feature_o = roi_pool(post_pmps1_feature_o, bbox_o, self.roi_size)
        post_pool_feature_o = post_pool_feature_o.view(post_pool_feature_o.size(0), -1)
        post_pool_feature_p = roi_pool(post_pmps1_feature_p, union_bbox(bbox_s, bbox_o), self.roi_size)
        post_pool_feature_p = post_pool_feature_p.view(post_pool_feature_p.size(0), -1)

        # gather
        pmps2_gather_s = F.relu(self.pmps2_gather_batchnorm_s(self.pmps2_gather_linear_so(post_pool_feature_s)))
        pmps2_gather_o = F.relu(self.pmps2_gather_batchnorm_o(self.pmps2_gather_linear_so(post_pool_feature_o)))
        pmps2_gather_p = F.relu(self.pmps2_gather_batchnorm_p(self.pmps2_gather_linear_p(post_pool_feature_p) + \
                                self.pmps2_linear_s2p(pmps2_gather_s) + self.pmps2_linear_o2p(pmps2_gather_o)))

        # broadcast
        pmps2_broadcast_p = F.relu(self.pmps2_broadcast_batchnorm_p(self.pmps2_broadcast_linear_p(pmps2_gather_p)))
        #pmps2_broadcast_s = F.relu(self.pmps2_broadcast_linear_so(pmps2_gather_s) + self.pmps2_linear_p2s(pmps2_broadcast_p))
        #pmps2_broadcast_o = F.relu(self.pmps2_broadcast_linear_so(pmps2_gather_o) + self.pmps2_linear_p2o(pmps2_broadcast_p))

        logits = self.fc(pmps2_broadcast_p)
        return torch.sum(logits * predicate, 1)

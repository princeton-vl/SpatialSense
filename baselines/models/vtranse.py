import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
import math


class VtransE(nn.Module):

    def __init__(self, phrase_encoder, visual_feature_size, predicate_embedding_dim, backbone='resnet18'):
        super(VtransE, self).__init__()

        self.visual_feature_size = visual_feature_size
        self.phrase_encoder = phrase_encoder

        self.backbone = models.__dict__[backbone](pretrained=True)
        self.backbone = nn.Sequential(self.backbone.conv1,
                                    self.backbone.bn1,
                                    self.backbone.relu,
                                    self.backbone.maxpool,
                                    self.backbone.layer1,
                                    self.backbone.layer2,
                                    self.backbone.layer3,
                                    self.backbone.layer4)


        self.scale_factor = nn.Parameter(torch.Tensor(3))
        nn.init.uniform_(self.scale_factor)
        
        self.linear1 = nn.Linear(visual_feature_size * visual_feature_size * 512, visual_feature_size * visual_feature_size * 64)
        self.batchnorm1 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)
        self.linear2 = nn.Linear(visual_feature_size * visual_feature_size * 512, visual_feature_size * visual_feature_size * 64)
        self.batchnorm2 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)

        feature_dim = 300 + 4 + visual_feature_size * visual_feature_size * 64
        self.W_o = nn.Linear(feature_dim, predicate_embedding_dim)
        self.W_s = nn.Linear(feature_dim, predicate_embedding_dim)
        self.W_p = nn.Linear(predicate_embedding_dim, 9)


    def forward(self, subj, obj, full_im, t_s, t_o, bbox_s, bbox_o, predicates):
        classeme_subj = self.phrase_encoder(subj)
        classeme_obj = self.phrase_encoder(obj)

        img_feature_map = self.backbone(full_im)
        subj_img_feature = []
        obj_img_feature = []
        for i in range(bbox_s.size(0)):
            bbox_subj = self.fix_bbox(7 * bbox_s[i], 7)
            bbox_obj = self.fix_bbox(7 * bbox_o[i], 7)
            subj_img_feature.append(F.upsample(img_feature_map[i : (i + 1), :, bbox_subj[0] : bbox_subj[1], bbox_subj[2] : bbox_subj[3]], self.visual_feature_size, mode='bilinear'))
            obj_img_feature.append(F.upsample(img_feature_map[i : (i + 1), :, bbox_obj[0] : bbox_obj[1], bbox_obj[2] : bbox_obj[3]], self.visual_feature_size, mode='bilinear'))
        subj_img_feature = torch.cat(subj_img_feature)
        obj_img_feature = torch.cat(obj_img_feature)
        subj_img_feature = subj_img_feature.view(subj_img_feature.size(0), -1)
        obj_img_feature = obj_img_feature.view(obj_img_feature.size(0), -1)
        subj_img_feature = F.relu(self.batchnorm1(self.linear1(subj_img_feature)))
        obj_img_feature = F.relu(self.batchnorm2(self.linear2(obj_img_feature)))

        x_s = torch.cat([classeme_subj * self.scale_factor[0], t_s * self.scale_factor[1], subj_img_feature * self.scale_factor[2]], 1)
        x_o = torch.cat([classeme_obj * self.scale_factor[0], t_o * self.scale_factor[1], obj_img_feature *  self.scale_factor[2]], 1)


        v_s = F.relu(self.W_s(x_s))
        v_o = F.relu(self.W_o(x_o))
        
        return torch.sum(self.W_p(v_o - v_s) * predicates, 1)


    def fix_bbox(self, bbox, size):
        new_bbox = [int(bbox[0]), min(size, int(math.ceil(bbox[1]))),
                    int(bbox[2]), min(size, int(math.ceil(bbox[3])))]
        assert 0 <= new_bbox[0] < new_bbox[1] <= size and 0 <= new_bbox[2] < new_bbox[3] <= size
        return new_bbox

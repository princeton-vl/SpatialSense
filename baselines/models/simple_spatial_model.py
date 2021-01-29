import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSpatialModel(nn.Module):
    def __init__(self, input_dim, feature_dim, predicate_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.batchnorm1 = nn.BatchNorm1d(feature_dim)
        self.linear2 = nn.Linear(input_dim, feature_dim)
        self.batchnorm2 = nn.BatchNorm1d(feature_dim)
        self.linear3 = nn.Linear(predicate_dim, feature_dim)
        self.batchnorm3 = nn.BatchNorm1d(feature_dim)
        self.linear4 = nn.Linear(feature_dim, feature_dim // 2)
        self.batchnorm4 = nn.BatchNorm1d(feature_dim // 2)
        self.linear5 = nn.Linear(feature_dim // 2, 1)

    def forward(self, subj, obj, predi):
        subj_feature = self.linear1(subj)
        subj_feature = self.batchnorm1(subj_feature)
        subj_feature = F.relu(subj_feature)

        obj_feature = self.linear2(obj)
        obj_feature = self.batchnorm2(obj_feature)
        obj_feature = F.relu(obj_feature)

        predicate_feature = self.linear3(predi)
        predicate_feature = self.batchnorm3(predicate_feature)
        predicate_feature = F.relu(predicate_feature)

        fused_feature = subj_feature + obj_feature + predicate_feature
        fused_feature = self.linear4(fused_feature)
        fused_feature = self.batchnorm4(fused_feature)
        fused_feature = F.relu(fused_feature)

        logit = self.linear5(fused_feature)

        return logit

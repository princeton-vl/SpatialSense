import torch
import torch.nn as nn
import pdb


class MaskedRegression(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.linear = nn.Linear((400 if opts.spatial else 0) + (600 if opts.appr else 0), 9)

    def forward(self, spatial_feature, appearance_feature, predicate):
        if appearance_feature is None:
            feature = spatial_feature
        elif spatial_feature is None:
            feature = appearance_feature
        else:
            feature = torch.cat([spatial_feature, appearance_feature], dim=1)
        logits = torch.masked_select(self.linear(feature), predicate)
        return logits

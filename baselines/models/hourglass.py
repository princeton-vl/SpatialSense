import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.layernorm1 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.layernorm2 = nn.LayerNorm((out_channels // 2, in_size, in_size))
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, inp):
        x = F.relu(self.layernorm1(self.conv1(inp)))
        x = F.relu(self.layernorm2(self.conv2(x)))
        x = self.conv3(x)
        return x + self.conv_skip(inp)



class Hourglass(nn.Module):

    def __init__(self, im_size, feature_dim):
        super().__init__()
        assert im_size == 1 or im_size % 2 == 0
        self.skip_resblock = ResidualBlock(feature_dim, feature_dim, im_size)
        if im_size > 1:
            self.pre_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)
            self.layernorm1 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.sub_hourglass = Hourglass(im_size // 2, feature_dim)
            self.layernorm2 = nn.LayerNorm((feature_dim, im_size // 2, im_size // 2))
            self.post_resblock = ResidualBlock(feature_dim, feature_dim, im_size // 2)


    def forward(self, x):
        up = self.skip_resblock(x)
        if x.size(-1) == 1:
            return up
        down = F.max_pool2d(x, 2)
        down = F.relu(self.layernorm1(self.pre_resblock(down)))
        down = F.relu(self.layernorm2(self.sub_hourglass(down)))
        down = self.post_resblock(down)
        down = F.upsample(down, scale_factor=2)
        return up + down


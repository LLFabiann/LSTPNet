import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers.create_act import create_act_layer

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class TceModule(nn.Module):
    def __init__(
            self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid',
            rd_ratio=1/8, rd_channels=None, rd_divisor=8, use_mlp=False):
        super(TceModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2

        rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
        self.rd_channels = rd_channels
        self.eca_conv_1 = nn.Sequential(
                                    nn.Conv1d(1, rd_channels, kernel_size=kernel_size, padding=padding, bias=True),
                        )
        self.action_p2_conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.relu = nn.ReLU()
        self.eca_conv_2 = nn.Sequential(
                                    # nn.ReLU(),
                                    nn.Conv1d(rd_channels, 1, kernel_size=kernel_size, padding=padding, bias=True),
                        )

        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b,c,h,w = x.shape
        
        max_out = self.maxpool(x).view(x.shape[0], 1, -1)
        avg_out = self.avg_pool(x).view(x.shape[0], 1, -1)
        max_out = self.eca_conv_1(max_out)
        max_out = max_out.view(-1, 16, self.rd_channels, c).permute(0,2,3,1).reshape(-1,c,16)
        max_out = self.action_p2_conv1(max_out).view(-1,self.rd_channels,c,16).permute(0,3,1,2).reshape(-1,self.rd_channels,c)
        max_out = self.relu(max_out)
        max_out = self.eca_conv_2(max_out).view(x.shape[0], -1, 1, 1)
        avg_out = self.eca_conv_1(avg_out)
        avg_out = avg_out.view(-1, 16, self.rd_channels, c).permute(0,2,3,1).reshape(-1,c,16)
        avg_out = self.action_p2_conv1(avg_out).view(-1,self.rd_channels,c,16).permute(0,3,1,2).reshape(-1,self.rd_channels,c)
        avg_out = self.relu(avg_out)
        avg_out = self.eca_conv_2(avg_out).view(x.shape[0], -1, 1, 1)
        y = max_out + avg_out
        y = self.sigmoid(y).reshape(-1,16,c,1,1)
        a2 = (y * 2).reshape(b,c,1,1).repeat(1,1,h,w)
        y = x * a2
        
        return y
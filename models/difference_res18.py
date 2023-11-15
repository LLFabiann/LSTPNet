from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from models.tce import TceModule

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockShift(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockShift, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.num_segments = num_segments
        self.eca = TceModule(channels = inplanes)

    def forward(self, x):
        residual = x

        x = self.eca(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, num_segments, block, layers, num_classes=7, end2end=True, num_of_embeddings=8):
        self.inplanes = 64
        self.end2end = end2end
        self.num_of_embeddings=num_of_embeddings
        self.num_segments = num_segments
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(self.num_segments, BasicBlock, 64, layers[0])
        
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], stride=2)
        
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], stride=2)
        
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2)
    
    def _make_layer(self, num_segments ,block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x_face=None, first_conv_feature=None, layer1_feature=None,  layer2_feature=None,
                                      layer3_feature=None):
        
        x = x.contiguous().view(-1, 3, 224, 224)
        # x = F.interpolate(x.contiguous().view(-1, 3, 112, 112), size=224)

        f = x # torch.Size([128, 3, 224, 224])
        
        f = self.conv1(f) # torch.Size([128, 64, 112, 112])
        f = self.bn1(f)
        f = self.relu(f)

        f = self.maxpool(f) # torch.Size([128, 64, 56, 56])
        f = self.layer1(f) # torch.Size([128, 64, 56, 56])
        
        f = self.layer2(f) # torch.Size([128, 128, 28, 28])

        f = self.layer3(f) # torch.Size([128, 256, 14, 14])
        
        f = self.layer4(f) # torch.Size([128, 512, 7, 7])
        
        return f

def resnet18(num_segments=16, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_segments, BasicBlockShift, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

if __name__ == '__main__':
    
    x = torch.ones(48,3,224,224)
    model = resnet18()
    output = model(x)
    print("test")


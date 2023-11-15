import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from models.tce import TceModule
from models.difference_res18 import resnet18

def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        
        # if k.startswith('backbone.'):
        #     k = k[9:]
        if k.startswith('module.'):
            k = k[7:]

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)

            # k = k.replace(k.split('.')[0], k.split('.')[0]+'_lbp')
            # new_state_dict[k] = v
            # matched_layers.append(k)

        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6

        if type == "large":
            depth = 0   

        self.img_size = img_size
        self.num_classes = num_classes

        self.ir_back = resnet18(num_classes=7)
        ir_checkpoint = torch.load('D:/DL_Study/Expression_Project/pre_train_models/resnet18_msceleb.pth', 
                                                                    map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.eca = TceModule(channels = 512)

    def forward(self, x):#torch.Size([16, 3, 224, 224])
        B_ = x.shape[0]
        x_ir = self.ir_back(x)#torch.Size([16, 49, 1024])
        x_fuse = x_ir
        x_fuse = self.eca(x_fuse)
        x_fuse = self.avgpool(x_fuse)
        x_fuse = torch.flatten(x_fuse, 1) # torch.Size([48, 512])
        return x_fuse
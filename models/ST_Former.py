import torch
from torch import nn
from models.LSTformer import temporal_transformer
from models.emotion_hyp_single_i_resnet18 import pyramid_trans_expr

class GenerateModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.s_former = pyramid_trans_expr()
        self.t_former = temporal_transformer(num_patches=16, dim=512, depth=2, heads=8, mlp_dim=4096, dim_head=64, dropout=0.0)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        
        x = self.s_former(x)
        x, atten_weight = self.t_former(x)
        features = x
        x = self.fc(x)
        
        return x, atten_weight, features
        


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)

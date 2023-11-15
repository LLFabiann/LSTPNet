import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math

atten_weight = torch.zeros(1,8,16,16).cuda()

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        in_features = dim
        hidden_features = hidden_dim
        out_features = dim

        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
        )
        self.TC = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_features)
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.TC(x) + x # + x
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1) # torch.Size([1, 8, 16, 16])
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        global atten_weight
        atten_weight = attn

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        
        self.TC = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.TC,
                self.norm,
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):

        for tc, norm, attn, ff in self.layers:
            x = x.transpose(1, 2)
            x = tc(x) + x # + x
            x = x.transpose(1, 2)
            x = norm(x)
            x = attn(x)
            x = ff(x)

        return x

class TFormer(nn.Module):
    def __init__(self, num_patches=16, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()
        
        self.num_patches = num_patches

        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):

        x = x.contiguous().view(-1, self.num_patches, 512) # torch.Size([3, 16, 512]) 
        
        x = self.spatial_transformer(x)

        x = x.mean(1)
        global atten_weight
        return x, atten_weight


def temporal_transformer(num_patches=16, dim=512, depth=2, heads=8, mlp_dim=4096, dim_head=64, dropout=0.0):
    return TFormer(num_patches=num_patches, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout)


if __name__ == '__main__':
    img = torch.randn(48,512)
    model = temporal_transformer()
    model(img)

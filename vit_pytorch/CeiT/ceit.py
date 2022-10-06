import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.CeiT.module import Residual, Attention, PreNorm, LeFF, FeedForward, LCAttention
import numpy as np
from vit_pytorch.face_losses import CosFace, ArcFace, SFaceLoss, Softmax


class TransformerLeFF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, scale=4, depth_kernel=3, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, LeFF(dim, scale, depth_kernel)))
            ]))

    def forward(self, x, mask=None):
        c = list()
        for attn, leff in self.layers:
            x = attn(x, mask=mask)
            cls_tokens = x[:, 0]
            c.append(cls_tokens)
            x = leff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        return x, torch.stack(c).transpose(0, 1)


class LCA(nn.Module):
    # I remove Residual connection from here, in paper author didn't explicitly mentioned to use Residual connection,
    # so I removed it, althougth with Residual connection also this code will work.
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(dim, LCAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x[:, -1].unsqueeze(1)
            x = x[:, -1].unsqueeze(1) + ff(x)
        return x


class CeiT(nn.Module):
    def __init__(self, *, image_size, loss_type, GPU_ID, patch_size, num_classes, dim=512, depth=20, heads=8,
                 pool='cls', in_channels=3,
                 out_channels=32, dim_head=64, dropout=0.,
                 emb_dropout=0., conv_kernel=7, stride=2, depth_kernel=3, pool_kernel=3, scale_dim=4, with_lca=False,
                 lca_heads=4, lca_dim_head=48, lca_mlp_dim=384):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # IoT
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, stride, 4),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel, stride)
        )
        print(self.conv)
        feature_size = image_size // 4
        print(feature_size)

        assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size // patch_size) ** 2
        patch_dim = out_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerLeFF(dim, depth, heads, dim_head, scale_dim, depth_kernel, dropout)

        self.with_lca = with_lca
        if with_lca:
            self.LCA = LCA(dim, lca_heads, lca_dim_head, lca_mlp_dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, num_classes)
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_classes, device_id=self.GPU_ID)

    def forward(self, img, label=None, mask=None):
        x = self.conv(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, c = self.transformer(x, mask)

        if self.with_lca:
            x = self.LCA(c)[:, 0]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

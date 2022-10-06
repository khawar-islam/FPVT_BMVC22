import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from flopth import flopth
from nystrom_attention import Nystromformer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory
# from apex.parallel import DistributedDataParallel as DDP

from config import get_config
# from get_flops import mha_flops
from image_iter import FaceDataset
from config import model_summary
from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy

import time
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import gc
# from torchsummary import summary
from vit_pytorch.Pit import PiT
from vit_pytorch.levit import LeViT

# from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

wandb.init(project='thesis', entity='khawar512')

gc.collect()
torch.cuda.empty_cache()

# Khawar
from vit_pytorch.cait import CaiT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.efficient_VIT import ViT
from x_transformers import Encoder

from vit_pytorch.dino import Dino
from vit_pytorch.vit import ViT
# from vit_pytorch.CvT.cvt import CvT

from vit_pytorch.t2t import T2TViT

# DEIT -FACEBOOK AI RESEARCH
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit_pytorch.nest import NesT

# from CrossViT.crossvit import CrossViT
teacher = resnet50(pretrained=True)

from vit_pytorch.CvT.cvt import CvT, ConvAttention
from vit_pytorch.CvT.module import ConvAttention, SepConv2d, FeedForward, Residual, PreNorm
from vit_pytorch.SwinT.swin import SwinTransformer
# Not working
from vit_pytorch.CCT.cct import CCT
from vit_pytorch.RVT.rvt import PoolingTransformer

# First Generation
varR = {'VIT': ViT_face(
    loss_type=HEAD_NAME,
    GPU_ID=GPU_ID,
    num_class=NUM_CLASS,
    image_size=112,
    patch_size=8,
    dim=512,
    depth=20,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
), 'VIT_base': ViT_face(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_class=NUM_CLASS,
        image_size=112,
        patch_size=8,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    ),
    'VITs': ViTs_face(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_class=NUM_CLASS,
        image_size=112,
        patch_size=8,
        ac_patch_size=12,
        pad=4,
        dim=512,
        depth=20,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ),
    # Optimization
    'CAiT': CaiT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_classes=NUM_CLASS,
        image_size=112,
        patch_size=8,
        ac_patch_size=12,
        cls_depth=2,
        pad=4,
        dim=512,
        depth=20,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ),
    # First Generation
    'DeepViT': DeepViT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        image_size=112,
        patch_size=8,
        num_classes=NUM_CLASS,
        ac_patch_size=12,
        pad=4,
        dim=512,
        heads=8,
        depth=20,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ),
    'RVT': PoolingTransformer(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_classes=NUM_CLASS,
        image_size=224,
        patch_size=16,
        stride=16,
        base_dims=[32, 32],
        depth=[10, 2],
        heads=[6, 12],
        mlp_ratio=4,
        use_mask=True,
        masked_block=10
    ),
    # Spatial Dimension
    'PiT': PiT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        image_size=112,
        ac_patch_size=12,
        num_classes=NUM_CLASS,
        patch_size=8,
        pad=4,
        dim=512,
        depth=(3, 3, 3),  # list of depths, indicating the number of rounds of each stage before a downsample
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ),
"PVT": PyramidVisionTransformer(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        img_size=112,
        patch_size=8,
        num_classes=NUM_CLASS,
        in_chans=3
    ),
    # Hirarchical (Local Attention)
    "NesT": NesT(
        loss_type='CosFace',
        GPU_ID=GPU_ID,
        image_size=224,
        patch_size=4,
        dim=96,
        heads=3,
        num_hierarchies=3,  # number of hierarchies
        block_repeats=(8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes=NUM_CLASS
    ),
"NesT_tiny": NesT(
        loss_type='CosFace',
        GPU_ID=GPU_ID,
        image_size=112,
        patch_size=14,
        dim=96,
        heads=3,
        num_hierarchies=3,  # number of hierarchies
        block_repeats=(2,2,8),  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes=NUM_CLASS
    ),
    # Hirarchical (Local Attention)
    "Swin": SwinTransformer(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=NUM_CLASS,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    ),
"Swin_Small": SwinTransformer(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        hidden_dim=96,
        layers=(2, 2, 18, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=NUM_CLASS,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
    # Second Generation
    "CvT": CvT(
        loss_type='CosFace',
        patch_size=8,
        ac_patch_size=12,
        pad=4,
        GPU_ID=GPU_ID,
        image_size=112,
        in_channels=3,
        num_classes=NUM_CLASS
    ),
# Seconf Generation
    "CeiT": CeiT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        image_size=112,
        patch_size=4,
        dim=512,
        depth=20,
        num_classes=NUM_CLASS,
        heads=8,
        dropout=0.1,
        emb_dropout=0.1
    ),
    # Token to Token
    "T2TViT": T2TViT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_classes=NUM_CLASS,
        image_size=224,
        # patch_size=8,
        # ac_patch_size=12,
        # pad=4,
        dim=512,
        depth=5,
        heads=8,
        mlp_dim=512,
        # dropout=0.1,
        # emb_dropout=0.1,
        t2t_layers=((7, 4), (3, 2), (3, 2))
        # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    ),
    "CCT": CCT(
        loss_type=HEAD_NAME,
        GPU_ID=GPU_ID,
        num_classes=NUM_CLASS,
        img_size=224,
        embedding_dim=384,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=14,
        num_heads=6,
        mlp_radio=3.,
        positional_embedding='learnable',  # ['sine', 'learnable', 'none']
    )
}


'''class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask_value = -torch.finfo(attn.dtype).max
        # embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x'''
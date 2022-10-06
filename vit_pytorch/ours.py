import torch
import torch.nn as nn
import torch.nn.functional as F
from Face_Loss.DictarcFace import DictArcMarginProduct
from vit_pytorch.face_losses import CosFace, MagFace, ArcFace, ArcMarginProduct, SFaceLoss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
# from torchvision.models import resnet34
# from Face_Loss.magface import MagFaceHeader
# import torch.optim as optim
# from vit_pytorch.localvit import LocalityFeedForward
from functools import partial
from torchvision.utils import save_image


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            # dim=256 coming from MLP block.
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            nn.BatchNorm2d(dim)
        )

        # self.dwconv = DeformableConv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
        # bias=True)

    def forward(self, x, H, W):
        # print(x.shape)  # x.shape=======>torch.Size([1, 784, 256]), H=28, W=28

        # B=1, C=256, B=784,
        B, N, C = x.shape
        # print(x.shape)  # torch.Size([1, 784, 256]), B=1,C=256,B=784

        # The view function is meant to reshape the tensor. x.shape==> torch.Size([1, 256, 28, 28])
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape)  # torch.Size([1, 256, 28, 28])

        # x.shape==> torch.Size([1, 256, 28, 28]), x.shape remain same before and after depthwise convolutions
        x = self.dwconv(x)  # torch.Size([1, 256, 28, 28])

        # print(x.shape)

        # x= x.flatten(2).shape===>torch.Size([1, 256, 784])
        # x.flatten(2).transpose(1, 2).shape==> torch.Size([1, 784, 256])
        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)

        # x = self.spatial_pyramid_pool(previous_conv_size=x.shape)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        # Coming from Block in_features=64, hidden_features=256
        # out_features is None the value of in_features has assigned to out_features
        out_features = out_features or in_features
        # hidden_features=256 coming from Block
        hidden_features = hidden_features or in_features

        # in_features=64, hidden_features=256
        self.fc1 = nn.Linear(in_features, hidden_features)

        # DW
        # (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        # (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()

        # hidden_features=256, out_features=64
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
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

    def forward(self, x, H, W):
        # H=28, W=28
        # x.shape===> torch.Size([1, 784, 64])

        x = self.fc1(x)
        # torch.Size([1, 784, 256])

        if self.linear:
            x = self.relu(x)

        x = self.dwconv(x, H, W)
        # print(x.shape)  # x.shape=>torch.Size([1, 784, 256])

        x = self.act(x)
        # print(x.shape)  # torch.Size([1, 784, 256])

        x = self.drop(x)
        # print(x.shape)  # torch.Size([1, 784, 256])

        x = self.fc2(x)  # torch.Size([1, 784, 64])
        # print(x.shape)

        x = self.drop(x)
        # print(x.shape)  # torch.Size([1, 784, 64])
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # print(self.dim)  # 64

        self.num_heads = num_heads
        # print(self.num_heads)  # 1

        head_dim = dim // num_heads
        # print(head_dim)  # 64

        self.scale = qk_scale or head_dim ** -0.5
        # print(self.scale)  # the value of qk_scale is "Empty",

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # print(self.q)  # Linear(in_features=64, out_features=64, bias=False)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # print(self.kv)  # Linear(in_features=64, out_features=128, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        # print(self.attn_drop)  # Dropout(p=0.0, inplace=False)

        self.proj = nn.Linear(dim, dim)
        # print(self.proj)  # Linear(in_features=64, out_features=64, bias=True)

        self.proj_drop = nn.Dropout(proj_drop)
        # print(proj_drop)  # 0.0

        self.linear = linear
        # print(self.linear)  # True

        self.sr_ratio = sr_ratio
        # print(self.sr_ratio)  # 8

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveMaxPool2d(7)
            # print(self.pool)  # AdaptiveMaxPool2d(output_size=7)

            # In average-pooling or max-pooling, you essentially set the stride and kernel-size by your own,
            # setting them as hyper-parameters. You will have to re-configure them if you happen to change your input
            # size.
            #
            # In Adaptive Pooling on the other hand, we specify the output size instead. And the stride and
            # kernel-size are automatically selected to adapt to the needs. The following equations are used to
            # calculate the value in the source code.

            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            # print(self.sr)  # Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

            self.norm = nn.LayerNorm(dim)
            # print(self.norm)  # LayerNorm((64,), eps=1e-05, elementwise_affine=True)

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
        # print(x.shape)  # torch.Size([1, 784, 64])
        # print(H, W)  # 28 28

        B, N, C = x.shape  # c=64,b=1,n=784

        # print(x.shape)  # torch.Size([1, 784, 64])
        # self.q= Linear(in_features=64, out_features=64, bias=False), self.num_heads=1,C // self.num_heads=64
        # self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).shape=====>torch.Size([1, 784, 1, 64])

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print(q.shape)  # torch.Size([1, 1, 784, 64])

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # print(x.shape)  # torch.Size([1, 784, 64])

            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # print(x_.shape)  # torch.Size([1, 64, 28, 28])

            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            # print(x_.shape)  # torch.Size([1, 49, 64])

            x_ = self.norm(x_)
            # print(x_.shape)  # torch.Size([1, 49, 64])

            x_ = self.act(x_)
            # print(x_.shape)  # torch.Size([1, 49, 64])

            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # print(kv.shape)  # torch.Size([2, 1, 1, 49, 64])

        k, v = kv[0], kv[1]
        # print(k.shape)  # torch.Size([1, 1, 49, 64])
        # print(v.shape)  # torch.Size([1, 1, 49, 64])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(attn.shape)  # torch.Size([1, 1, 784, 49])

        mask_value = -torch.finfo(attn.dtype).max
        # embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = attn.softmax(dim=-1)
        # print(attn.shape)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(x.shape)

        x = self.proj(x)
        # print(x.shape)

        x = self.proj_drop(x)
        # print(x.shape)  # torch.Size([1, 784, 64])

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Debugging
        # dim=64, mlp_ratio=4, dim * mlp_ratio=256
        mlp_hidden_dim = int(dim * mlp_ratio)  #
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

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

    def forward(self, x, H, W):
        # print(x.shape, H, W)  # torch.Size([1, 784, 64]) 28 28

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # print(x.shape)  # torch.Size([1, 784, 64])

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))  # here we give H,W to mlp block
        # print(x.shape)  # torch.Size([1, 784, 64])

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=112, patch_size=8, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        # print("Image Size:", img_size)  # (112, 112)

        patch_size = to_2tuple(patch_size)
        # print("Patch Size:", patch_size)  # (7, 7)

        self.img_size = img_size
        # print("Image Size:", self.img_size)  # (112, 112)

        self.patch_size = patch_size
        # print("Patch Size:", self.patch_size)  # (7, 7)

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # print(self.H, self.W, img_size[0], img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 16 16 112
        # 16 16

        self.num_patches = self.H * self.W
        # print("Number of WxH:", self.num_patches)  # 256

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # print(self.proj)  # Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, x):
        # print(x.shape)  # torch.Size([1, 3, 112, 112])

        x = self.proj(x)
        # print(x.shape)  # torch.Size([1, 64, 28, 28])

        _, _, H, W = x.shape
        # print(_, _, H, W)  # 64 64 28 28

        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)  # torch.Size([1, 784, 64])

        x = self.norm(x)
        # print(x.shape)  # torch.Size([1, 784, 64])

        return x, H, W


class Ours_FPVT(nn.Module):
    def __init__(self, *, img_size=112, patch_size=8, loss_type, GPU_ID, in_chans=3, num_classes=526,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=True):

        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        # print("self.patch_embed")
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=8, stride=8, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=3, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=3, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        # print("self.patch_embed4")
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=3, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], linear=linear)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], linear=linear)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], linear=linear)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dims[3]),
        )

        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=embed_dims[3], out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=embed_dims[3], out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=embed_dims[3], out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=embed_dims[3], out_features=num_classes, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcMarginProduct':
                self.loss = ArcMarginProduct(in_features=embed_dims[3], out_features=num_classes, m=0.1)
                # self.fc2 = DictArcMarginProduct(embed_dims[3], out_features=num_classes, out_features_test=num_classes, label_dict=None, m=0.5)

        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    #
    def forward_features(self, x):
        """Returns a contiguous tensor containing the same data as self tensor. If self tensor is contiguous,
        this function returns the self tensor. """
        B = x.shape[0]

        # print(x.shape)
        # torch.Size([1, 3, 112, 112])



        '''##########################################  stage 1'''
        x, H, W = self.patch_embed1(x)
        #print(x.shape)

        # torch.Size([1, 784, 64]), H=28, W=28

        # print(H, W)
        # 28 28

        for i, blk in enumerate(self.block1):
            # print(x.shape)  # torch.Size([1, 784, 64])
            x = blk(x, H, W)
        x = self.norm1(x)
        # print(x.shape)
        # torch.Size([1, 784, 64])

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(x.reshape(B, H, W, -1).shape)
        # torch.Size([1, 28, 28, 64])
        # print(x.shape)
        # torch.Size([1, 64, 28, 28])
        # save_image(x, 'img' + str(1) + '.png')

        '''##########################################  stage 2'''
        # print(x.shape)
        # torch.Size([1, 64, 28, 28])

        x, H, W = self.patch_embed2(x)
        # print("stage 2", x.shape)
        # torch.Size([1, 196, 128])
        # print("stage 2", H, W)
        # 14 14

        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
            # print("stage 2", x.shape)
            # torch.Size([1, 196, 128])
        x = self.norm2(x)
        # print("stage 2", x.shape)
        # torch.Size([1, 196, 128])

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print("stage 2", x.shape)
        # torch.Size([1, 128, 14, 14])

        '''################################################### stage 3'''
        # print(x.shape)
        # torch.Size([1, 128, 14, 14])

        x, H, W = self.patch_embed3(x)
        # print("stage 3:", x.shape)
        # stage 3 torch.Size([1, 49, 256])
        # print(H, W)
        # 7 7

        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        # print("self.norm3(x):", x.shape)  # self.norm3(x): torch.Size([1, 49, 256])
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print("stage 3:", x.shape)
        # stage 3: torch.Size([1, 256, 7, 7])

        '''#################################################### stage 4'''
        # print("stage 4:", x.shape)
        # stage 4: torch.Size([1, 256, 7, 7])

        x, H, W = self.patch_embed4(x)
        # print("stage 4:", x.shape)
        # stage 4: torch.Size([1, 16, 512])

        # print("stage 4:", H, W)
        # H=4, W=4

        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x.mean(dim=1)

    def forward(self, x, label=None, mask=None):
        x = self.forward_features(x)
        x = self.mlp_head(x)

        if label is not None:
            loss_value = self.loss(x, label)
            return loss_value, x
        else:
            return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_v2_b0(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    model = Ours_FPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model

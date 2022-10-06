import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.face_losses import CosFace, ArcFace, SFaceLoss, Softmax

MIN_NUM_PATCHES = 16
class ViT(nn.Module):
    def __init__(self, *, image_size, loss_type, GPU_ID, pad=4, ac_patch_size, patch_size, num_classes, dim, transformer, pool='cls', channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size),
                                    stride=(self.patch_size, self.patch_size), padding=(pad, pad))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            #nn.Linear(dim, num_classes)
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
        p = self.patch_size
        x = self.soft_split(img).transpose(1, 2)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)

        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

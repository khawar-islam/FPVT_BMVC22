import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model
from PIL import Image

model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
model = create_model(model_name, pretrained=False).to(device)

IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
]

transforms = T.Compose(transforms)

# Demo Image
img = PIL.Image.open('Adam_Brody_233.png')
#img = img.resize((224, 224), Image.ANTIALIAS)
img_tensor = transforms(img).unsqueeze(0).to(device)

# end-to-end inference
output = model.patch_embed(img_tensor)

print("Inference Result:")
print("Face")
plt.imshow(img)


# 1. Split Image into Patches The input image is split into N patches (N = 14 x 14 for ViT-Base) and converted to
# D=768=16x16x3 embedding vectors by learnable 2D convolution: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

patches = model.patch_embed(img_tensor)  # patch embedding convolution

print("Image tensor: ", img_tensor.shape)
# Image tensor:  torch.Size([1, 3, 224, 224])

print("Patch embeddings: ", patches.shape)
# Patch embeddings:  torch.Size([1, 196, 768])

# This is NOT a part of the pipeline.
# Actually the image is divided into patch embeddings by Conv2d
# with stride=(16, 16) shown above.
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of Patches", fontsize=24)
fig.add_axes()
img = np.asarray(img)
for i in range(0, 196):  # 14 x14 (number of patches in width and height)
    x = i % 14
    y = i // 14
    patch = img[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
    ax = fig.add_subplot(14, 14, i + 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)

plt.savefig('ddd.png')


# https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=J3GovnsM1t0f
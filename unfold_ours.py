# import torch
# import numpy as np
# import torch.nn as nn
# from torch import nn
# from PIL import Image
# import cv2
# import os
# import math
# import torch.nn.functional as F
# import torchvision.transforms as T
# from timm import create_model
# from typing import List
#
# import matplotlib.pyplot as plt
# from torchvision import io, transforms
# from utils_torch import Image, ImageDraw
# from torchvision.transforms.functional import to_pil_image
#
# IMG_SIZE = 112
# # PATCH_SIZE = 64
#
# resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
# img = resize(io.read_image("Adam_Brody_233.png"))
#
# image_size = 112
# patch_size = 8
# ac_patch_size = 12
# pad = 4
#
# patches = img.unfold(1, ac_patch_size, patch_size).unfold(2, ac_patch_size, patch_size)
# print(patches.shape)
#
#
# fig, ax = plt.subplots(12, 12)
# for i in range(12):
#     for j in range(12):
#         sub_img = patches[:, i, j]
#         ax[i][j].imshow(to_pil_image(sub_img))
#         ax[i][j].axis('off')
#
# plt.show()


import torch
import numpy as np
import torch.nn as nn
from torch import nn
from PIL import Image
import cv2
import os
import math
import torch.nn.functional as F
import torchvision.transforms as T
from timm import create_model
from typing import List

import matplotlib.pyplot as plt
from torchvision import io, transforms
from utils_torch import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

IMG_SIZE = 112
# PATCH_SIZE = 64

resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
img = resize(io.read_image("Adam_Brody_233.png"))
img = img.to(torch.float32)

image_size = 112
patch_size = 28
ac_patch_size = 12
pad = 4

img = img.unsqueeze(0)
soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(patch_size, patch_size), padding=(pad, pad))
patches = soft_split(img).transpose(1, 2)

fig, ax = plt.subplots(16, 16)
for i in range(16):
    for j in range(16):
        sub_img = patches[:, i, j]
        ax[i][j].imshow(to_pil_image(sub_img))
        ax[i][j].axis('off')

plt.show()
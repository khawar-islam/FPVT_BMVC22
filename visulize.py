import torch
import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json

from vit_pytorch.ours import Ours_FPVT
from vit_pytorch.vits_face import ViTs_face

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('/home/cvpr/Documents/OPVT/skin-original.jpg'))
plt.imshow(image)

model = Ours_FPVT(
    loss_type='ArcMarginProduct',
    GPU_ID=0,
    patch_size=8,
    img_size=112,
    depths=[3, 4, 18, 3],
    num_classes=526,
    in_chans=3
)
model.load_state_dict(torch.load('/home/cvpr/Documents/OPVT/results/FPVT_112_8_8/Backbone_PVTV2_Epoch_1_Batch_700_Time_2022-07-07-12-08_checkpoint.pth'))

#model = models.resnet18(pretrained=True)
print(model)

# we will save the conv layer weights in this list
model_weights = []
# we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if model_children[i] == model.patch_embed1:
        counter += 1
        weigh = model_children[i].proj
        model_weights.append(weigh.weight)
        conv_layers.append(model_children[i].proj)
    elif model_children[i] == model.patch_embed2:
        counter += 1
        weigh = model_children[i].proj
        model_weights.append(weigh.weight)
        conv_layers.append(model_children[i].proj)
    elif model_children[i] == model.patch_embed3:
        counter += 1
        weigh = model_children[i].proj
        model_weights.append(weigh.weight)
        conv_layers.append(model_children[i].proj)
    elif model_children[i] == model.patch_embed4:
        counter += 1
        weigh = model_children[i].proj
        model_weights.append(weigh.weight)
        conv_layers.append(model_children[i].proj)


print(f"Total convolution layers: {counter}")
print("conv_layers")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)


outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
#print feature_maps
for feature_map in outputs:
    print("")
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(50, 50))
for i in range(len(processed)):
    a = fig.add_subplot(2, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=20)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
plt.show()
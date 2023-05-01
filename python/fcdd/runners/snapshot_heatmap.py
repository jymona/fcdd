import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from fcdd.datasets.image_folder import ImageFolder
from fcdd.datasets.preprocessing import local_contrast_normalization
from fcdd.training.fcdd import FCDDTrainer
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F
from fcdd.datasets.image_folder import ADImageFolderDataset

min_max_l1 = [
    [(-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
     (1.3779616355895996, 1.3779616355895996, 1.3779616355895996)],
    [(-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
     (4.573435306549072, 4.573435306549072, 4.573435306549072)],
    [(-3.184587001800537, -3.164201259613037, -3.1392977237701416),
     (1.6995097398757935, 1.6011602878570557, 1.5209171772003174)],
    [(-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
     (6.503103256225586, 5.875098705291748, 5.814228057861328)],
    [(-3.100773334503174, -3.100773334503174, -3.100773334503174),
     (4.27892541885376, 4.27892541885376, 4.27892541885376)],
    [(-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
     (18.966819763183594, 21.64590072631836, 26.408710479736328)],
    [(-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
     (11.564697265625, 10.976534843444824, 10.378695487976074)],
    [(-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
     (6.854909896850586, 6.854909896850586, 6.854909896850586)],
    [(-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
     (2.290637969970703, 2.4007883071899414, 2.3044068813323975)],
    [(-7.236185073852539, -7.236185073852539, -7.236185073852539),
     (3.3777384757995605, 3.3777384757995605, 3.3777384757995605)],
    [(-3.2036616802215576, -3.221003532409668, -3.305514335632324),
     (7.022546768188477, 6.115569114685059, 6.310940742492676)],
    [(-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
     (4.4255571365356445, 4.642300128936768, 4.305730819702148)],
    [(-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
     (5.463134765625, 5.463134765625, 5.463134765625)],
    [(-2.9547364711761475, -3.17536997795105, -3.143850803375244),
     (5.305514812469482, 4.535006523132324, 3.3618252277374268)],
    [(-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
     (2.515115737915039, 2.515115737915039, 2.515115737915039)]
]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    # print(img.shape)
    img = np.float32(img) / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, requires_grad=True)
    # img = img.unsqueeze(0)
    print(img.shape)
    return img


def preprocess_image2(image_path, device):
    raw_shape = (3, 248, 248)
    shape = (3, 224, 224)
    transform = transforms.Compose([
        transforms.Resize(raw_shape[-1]),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(shape[-1]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        transforms.Normalize(min_max_l1[0][0],
        [ma - mi for ma, mi in zip(min_max_l1[0][1], min_max_l1[0][0])])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批量维度
    img_tensor = img_tensor.to(device)
    return img_tensor

def get_features(img, model, layer_num):
    features = img
    for index, layer in enumerate(model.features):
        features = layer(features)
        if index == layer_num:
            break
    return features


def visualize_heatmap(feature_maps):
    feature_maps_np = feature_maps.cpu().detach().numpy().squeeze()
    feature_maps_np = np.mean(feature_maps_np, axis=0)
    heatmap = (feature_maps_np - np.min(feature_maps_np)) / (np.max(feature_maps_np) - np.min(feature_maps_np))
    print(heatmap.shape)
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.show()


# Path to your snapshot.pt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

snapshot = "/home/dulab/Project/fcdd/python/data/results/fcdd_CF/normal_0/it_0/snapshot.pt"
state_dict = torch.load(snapshot)
logger = None  # Logger("fcdd/data/results/foo")
quantile = 0.97
net = FCDD_CNN224_VGG_F((3, 224, 224), bias=True).cuda()
net.load_state_dict(state_dict['net'])
net.eval()
image_path = "/home/dulab/Project/fcdd/python/data/datasets/custom/test/cf_medium/120.png"
layer_num = 0

input_image = preprocess_image2(image_path, device)
conv_features = get_features(input_image, net, layer_num)
visualize_heatmap(conv_features)


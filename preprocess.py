import torch
import torch.nn as nn
from torch.utils.data import Datasetv, DataLoader
import os
import tqdm
from glob import glob
from PIL import Image

import monai
from monai.data import create_test_image_2d, decollate_batch, DataLoader, Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    ScaleIntensityRanged,
    SpatialPadd,
    Resized,
    ToTensord,
)
from monai.visualize import plot_2d_or_3d_image

img_dir = "C:\Users\CogPro VR-5\Desktop\vistibular\UNET\UNET\images"
mask_dir = "C:\Users\CogPro VR-5\Desktop\vistibular\UNET\UNET\masks"

data = {}

# ensuring same formatting of data across various os
for i in sorted(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, i)
    mask_path = os.path.join(mask_dir, i)
    if not os.path.exists(mask_path):
        print(f"Missing Mask : {mask_path}")
    data.append({"image" : img_path, "mask" : mask_path})

# to transform data into constant format(size, scale)
img_transform = Compose(
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstD(keys=["image"]), #ensure CHW, expected channel dimension remains constant, C->channels, H->Height, W->Width 
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max = 255, b_min = 0.0, b_max=1.0),
    SpatialPadd(keys=["image"], spatial_size=(1920,1920), method='symmetric'),
    Resized(keys=["image"], spatial_size=(256,256), mode=["area", "nearest"]),
    ToTensord()
)

mask_transform = Compose(
    LoadImaged(keys=["mask"], image_only=True),
    EnsureChannelFirstD(keys=["mask"]), #ensure CHW, expected channel dimension remains constant, C->channels, H->Height, W->Width 
    # ScaleIntensityRanged(keys=["mask"], a_min=0, a_max = 255, b_min = 0.0, b_max=1.0),
    SpatialPadd(keys=["mask"], spatial_size=(1920,1920), method='symmetric'),
    Resized(keys=["mask"], spatial_size=(256,256), mode=["area", "nearest"]),
    ToTensord()
)
class IrisDataset(Dataset):
    def __init__(self, data, img_transform, mask_transform):
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        image = self.img_transform[item["image"]]
        mask = self.mask_transform[item["mask"]]
        return image, mask

dataset = IrisDataset(data, img_transform, mask_transform)
loader = DataLoader(dataset, batch_size = , shuffle = True)
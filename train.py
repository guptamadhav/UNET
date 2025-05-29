import os
from glob import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor, Compose
)

from monai.networks.nets import UNet
from tqdm import tqdm

image_dir = "pupilDataset/train_img"  # 🔁 Your images folder
mask_dir = "pupilDataset/train_mask"    # 🔁 Your masks folder
image_size = (256, 256)
batch_size = 4
num_epochs = 10
lr = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")


image_filenames = sorted(os.listdir(image_dir))
data = []
for f in sorted(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, f)
    mask_path = os.path.join(mask_dir, f)

    if not os.path.exists(mask_path):
        print(f"❌ Missing mask: {mask_path}")
        continue

    data.append({"image": img_path, "mask": mask_path})


image_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),       
    ScaleIntensity(),
    Resize((128,128)),
    ToTensor()
])
mask_transforms = Compose([
    LoadImage(image_only=True),                  
    EnsureChannelFirst(), # [1,H,W]
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((128,128), mode="nearest"),  # NEAREST keeps 0/1 labels crisp
    ToTensor(),                 # now exactly 0.0 or 1.0
])
# -------------------------------
# CUSTOM DATASET
# -------------------------------
class PupilDataset(Dataset):
    def __init__(self, data, image_transform, mask_transform):
        self.data = data
        self.image_transform = image_transform
        self.mask_transform = mask_transform
    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.image_transform(item["image"])
        mask = self.mask_transform(item["mask"])
        return image, mask

    def __len__(self):
        return len(self.data)

dataset = PupilDataset(data, image_transforms, mask_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# MONAI U-NET
# -------------------------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16,32,64,128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

# -------------------------------
# LOSS + OPTIMIZER
# -------------------------------
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# TRAINING LOOP
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"✅ Epoch {epoch+1}: Avg Loss = {epoch_loss / len(loader):.4f}")

# -------------------------------
# SAVE MODEL
# -------------------------------
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "opt_state":  optimizer.state_dict(),
},"unet_pupil_segmentation.pth")

print("🎉 Model saved as unet_pupil_segmentation.pth")

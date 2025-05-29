import os, torch, numpy as np
from glob import glob
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# 1.  paths
# ------------------------------------------------------------------
test_img_dir  = "/Users/madhav/Desktop/UNET/pupilDataset/train_img"     # all unseen images
test_mask_dir = "/Users/madhav/Desktop/UNET/pupilDataset/train_mask"    # matching GT masks
model_path    = "/Users/madhav/Desktop/UNET/unet_pupil_segmentation.pth"

img_tf = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),        # -> (1,H,W)
    ScaleIntensity(),
    Resize((128, 128)),
    ToTensor()
])
mask_tf = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((128, 128), mode="nearest"),
    ToTensor()                   # still 0. or 1.
])

class TestSet(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*")))
        self.msk_dir   = msk_dir
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        fname     = os.path.basename(img_path)
        mask_path = os.path.join(self.msk_dir, fname)
        img  = img_tf(img_path)
        mask = mask_tf(mask_path)
        return img, mask

loader = DataLoader(TestSet(test_img_dir, test_mask_dir),
                    batch_size=4, shuffle=False)

# ------------------------------------------------------------------
def load_model(model_path, device):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16,32,64,128),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    
    # Remove "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True) 
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                 else "cuda" if torch.cuda.is_available()
                 else "cpu")

model  = load_model(model_path, device)   # ← your function


dice_metric = DiceMetric(
    include_background=False,   # we only care about the foreground (pupil)
    reduction="none",
    ignore_empty=True,          # ← skip 0–mask + 0–prediction pairs
    get_not_nans=True
)

# ------------------------------------------------------------------
# 5.  evaluation loop
# ------------------------------------------------------------------
model.eval()
all_dice = []

with torch.no_grad():
    for imgs, gts in loader:
        imgs, gts = imgs.to(device), gts.to(device)

        logits  = model(imgs)
        probs   = torch.sigmoid(logits)
        preds   = (probs > 0.6).float()

        # dice_metric returns tensor(shape=batch)   e.g. [0.87, 0.90, …]
        batch_dice = dice_metric(preds, gts)
        all_dice.extend(batch_dice.cpu().tolist())

mean_dice = np.mean(all_dice)
print(f"\n🔍  Evaluated {len(all_dice)} images — Mean Dice = {mean_dice:.4f}")

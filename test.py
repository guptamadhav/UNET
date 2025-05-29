import torch
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# --- Replace with your actual model class or import ---
from monai.networks.nets import UNet

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
            new_key = k[7:]  # remove "module." prefix
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)  # or strict=False to skip missing keys
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),       
        ScaleIntensity(),
        Resize((128,128)),
        ToTensor()
    ])
    img = transforms(image_path)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def predict_mask(model, input_tensor, device):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.6).float()
    return pred_mask.cpu().squeeze().numpy()

def save_mask(mask_np, output_path):
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img.save(output_path)

def display_mask(mask_np):
    plt.imshow(mask_np, cmap='gray')
    plt.title('Predicted Pupil Segmentation Mask')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
    model_path = "/Users/madhav/Desktop/UNET/unet_pupil_segmentation.pth"  # <-- set your model file path here
    image_path = "/Users/madhav/Desktop/UNET/pupilDataset/test_img/IRD_2955_1.png"    # <-- set your input image path here
    output_mask_path = "predicted_mask.png"

    model = load_model(model_path, device)
    input_tensor = preprocess_image(image_path)
    mask_np = predict_mask(model, input_tensor, device)

    display_mask(mask_np)
    save_mask(mask_np, output_mask_path)
    print(f"Mask saved to {output_mask_path}")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ======== Parameters ========
# Make sure this matches your data directory structure
test_patches_dir = "D:\Project\Coconut Plantation\Applicatno\patches_split"
if os.path.exists(test_patches_dir):
    num_classes = len(os.listdir(test_patches_dir))
else:
    num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Define Transforms ========
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======== Load Model (MobileNet V2) ========
model = models.mobilenet_v2(weights=None) # No weights needed, we load our own
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load your trained weights
if os.path.exists("mobilenet_v2_density_classifier.pth"):
    model.load_state_dict(torch.load("mobilenet_v2_density_classifier.pth", map_location=device))
    print("MobileNet V2 weights loaded.")
    
model = model.to(device)
model.eval()

# ======== Inference ========
# Example image path (update this)
img_path = "patches_split/test/class 5/SA-1-C-4.png"

if os.path.exists(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        probs = probs.cpu().numpy().flatten()

    for i, prob in enumerate(probs):
        print(f"Class {i}: {prob:.4f}")
else:
    print(f"Image not found at {img_path}")
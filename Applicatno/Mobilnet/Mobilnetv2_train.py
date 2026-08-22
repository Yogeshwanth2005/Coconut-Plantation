import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ======== Parameters ========
data_dir = "D:\Project\Coconut Plantation\Applicatno\patches_split"
# Ensure the directory exists or change path
if os.path.exists(data_dir):
    num_classes = len(os.listdir(data_dir))
else:
    num_classes = 2 # Placeholder if path doesn't exist locally

batch_size = 16
num_epochs = 20
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Transforms (Same as ResNet) ========
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======== Dataset & DataLoader ========
# Only run if directory exists to avoid errors in dry run
if os.path.exists(data_dir):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
else:
    print("Data directory not found. Please check 'data_dir'.")

# ======== Model Setup: MobileNet V2 ========
# Using 'DEFAULT' weights which are the best available ImageNet weights
model = models.mobilenet_v2(weights='DEFAULT')

# MobileNetV2 classifier is a Sequential block: (0): Dropout, (1): Linear
# We replace the last Linear layer at index 1
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ======== Loss and Optimizer ========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======== Tracking Lists ========
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# ======== Training Loop ========
if os.path.exists(data_dir):
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0, 0, 0

        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in prog_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            prog_bar.set_postfix(loss=loss.item())

        train_acc  = correct / total
        train_loss = total_loss / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

        # --- Validate ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            prog_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in prog_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc  = correct / total
        val_loss = val_loss / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"           Validation Loss: {val_loss:.4f}  Validation Acc: {val_acc:.4f}")

    # ======== Plot Curves ========
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MobileNet V2 Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies,   label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MobileNet V2 Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ======== Save Model ========
    torch.save(model.state_dict(), "mobilenet_v2_density_classifier.pth")
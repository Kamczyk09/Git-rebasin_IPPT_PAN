import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trening modelu Resnet18 na obrazach biomedycznych (zapalenie płuc)
def evaluate(model):
    data_dir = "matching_mod/data/medical_images"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

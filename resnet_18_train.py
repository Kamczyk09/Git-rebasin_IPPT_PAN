import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters:
lr = 0.005
nEpochs = 5
batch_size = 128

data_dir = "matching_mod/data/medical_images"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Mapowanie klas:", train_dataset.class_to_idx)


model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 1)
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

train_losses = []
test_losses = []
mccs = []
for epoch in range(nEpochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss_train = running_loss / len(train_loader)
    train_losses.append(epoch_loss_train)

    running_loss = 0.0
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss_test = running_loss / len(test_loader)
    test_losses.append(epoch_loss_test)

    all_preds = np.vstack(all_preds).flatten()
    all_labels = np.vstack(all_labels).flatten()

    mcc = matthews_corrcoef(all_labels, all_preds)
    mccs.append(mcc)

    print(
        f"Epoch [{epoch+1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss test: {epoch_loss_test:.4f}, MCC: {mcc:.4f}")

torch.save(model.state_dict(), "przykładowe_wyniki/resnet18.pth")

fig, ax = plt.subplots(1,2)
ax[0].plot(train_losses, label="Train")
ax[0].plot(test_losses, label="Test")
ax[0].legend()

ax[1].plot(mccs)

plt.savefig("resnet_18_train.png")

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += preds.eq(labels.unsqueeze(1)).sum().item()
        total += labels.size(0)

print("Accuracy: {:.2f}%".format(100 * correct / total))



    



import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters:
lr = 0.001
nEpochs = 20
batch_size = 64

data_dir = "./medical_images"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Klasy:", train_dataset.classes)
print("Mapowanie klas:", train_dataset.class_to_idx)


model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

train_losses = []
test_losses = []
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

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch, nEpochs, epoch_loss))

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

        epoch_loss = running_loss / len(test_loader)
        test_losses.append(epoch_loss)

torch.save(model.state_dict(), "resnet18.pth")

fig, ax = plt.subplots()
ax.plot(train_losses, label="Train")
ax.plot(test_losses, label="Test")
ax.legend()

plt.savefig("resnet_18_train.png")
plt.show()

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



    



import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.medical_images import load_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(nOutputNeurons, nEpochs, pretrained=False):

    # hyperparameters:
    lr = 0.005
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test = load_data(pretrained=pretrained)
    #############################

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    print("Mapowanie klas:", dataset_test.class_to_idx)

    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, nOutputNeurons)
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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

        scheduler.step()

        epoch_loss_train = running_loss / len(train_loader)
        train_losses.append(epoch_loss_train)

        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        epoch_loss_test = running_loss / len(test_loader)
        test_losses.append(epoch_loss_test)

        print(
            f"Epoch [{epoch+1}/{nEpochs}], Loss train: {epoch_loss_train:.4f}, Loss test: {epoch_loss_test:.4f}")

    torch.save(model.state_dict(), f'matching_mod/models/resnet18_new{"_pretrained" if pretrained else "raw"}.pth')

    fig, ax = plt.subplots(1,1)
    ax.plot(train_losses, label="Train")
    ax.plot(test_losses, label="Test")
    ax.legend()

    plt.savefig(f"resnet_18{'_pretrained' if pretrained else '_raw'}.png")

    return model.state_dict()


def evaluate(model, pretrained=False):
    batch_size = 128

    #####LOADING DATA############
    dataset_train, dataset_test = load_data(pretrained=pretrained)
    #############################

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
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


def return_model(nOutputNeurons):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, nOutputNeurons)
    )
    return model
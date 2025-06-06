from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
import os


def load_data(pretrained=False):

    if pretrained:
        transform = ResNet18_Weights.DEFAULT.transforms()
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "medical_images")

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    return train_dataset, test_dataset


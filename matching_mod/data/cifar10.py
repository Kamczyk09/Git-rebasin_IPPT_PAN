from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights

def load_data():

    # transform = ResNet18_Weights.DEFAULT.transforms()
    # # else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

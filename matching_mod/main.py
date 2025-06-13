import ResNet18
import torch

# print("Model raw training")
ResNet18.train(10, pretrained=True)
ResNet18.evaluate(pretrained=True)


import ResNet18
import CNN
import MLP
import torch

print("Model pretrained training")
MLP.train(15, pretrained=True)
MLP.evaluate(pretrained=True)

print("Model raw training")
MLP.train(10, pretrained=False)
MLP.evaluate(pretrained=False)


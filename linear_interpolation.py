import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from interpolation_model_eval import evaluate
import matplotlib.pyplot as plt

model = models.resnet18()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 1)
)

state_dict_1 = torch.load('przykładowe_wyniki/resnet18.pth', weights_only=True)
state_dict_2 = torch.load('przykładowe_wyniki/resnet18_pretrained.pth', weights_only=True)

alphas = np.linspace(0, 1, 60)

accs = []
for i, alpha in enumerate(alphas):
    print(f"Alpha: {i+1}")
    interpolated_state_dict = {}

    for key in state_dict_1.keys():
        interpolated_state_dict[key] = (1 - alpha) * state_dict_1[key] + alpha * state_dict_2[key]

    model.load_state_dict(interpolated_state_dict)
    accuracy = evaluate(model)
    accs.append(accuracy)

plt.plot(alphas, accs)
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('linear_interpolation.png')
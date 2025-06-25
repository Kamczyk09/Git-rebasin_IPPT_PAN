import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ResNet18 import evaluate, return_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict_1 = torch.load('models_checkpoints/resnet18_T800.pth', weights_only=True)
state_dict_2 = torch.load('models_checkpoints/resnet18_T1000.pth', weights_only=True)

alphas = np.linspace(0, 1, 50)

accs = []
for i, alpha in enumerate(alphas):
    print(f"Alpha: {i+1}")
    interpolated_state_dict = {}

    for key in state_dict_1.keys():
        interpolated_state_dict[key] = ((1 - alpha) * state_dict_1[key].float() + alpha * state_dict_2[key].float())

    model = return_model(10)
    model.load_state_dict(interpolated_state_dict)
    model.to(device)
    model.eval()
    accuracy = evaluate(model, pretrained=True)

    accs.append(accuracy)

plt.plot(alphas, accs)
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('linear_interpolation.png')
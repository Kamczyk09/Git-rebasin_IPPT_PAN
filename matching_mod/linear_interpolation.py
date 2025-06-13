import torch
import numpy as np
import matplotlib.pyplot as plt
import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict_1 = torch.load('models_checkpoints/resnet18_raw.pth', weights_only=True)
state_dict_2 = torch.load('models_checkpoints/resnet18_pretrained.pth', weights_only=True)

alphas = np.linspace(0, 1, 50)

accs = []
for i, alpha in enumerate(alphas):
    print(f"Alpha: {i+1}")
    interpolated_state_dict = {}

    for key in state_dict_1.keys():
        interpolated_state_dict[key] = ((1 - alpha) * state_dict_1[key].float() + alpha * state_dict_2[key].float())

    model = ResNet18.return_model(10)
    model.load_state_dict(interpolated_state_dict)
    model.to(device)
    model.eval()
    accuracy = ResNet18.evaluate(model, pretrained=True)

    accs.append(accuracy)

# model1 = ResNet18.return_model(10)
# model1.load_state_dict(state_dict_1)
# model1.eval()
# print("Acc raw:", ResNet18.evaluate(model1))
#
# model2 = ResNet18.return_model(10)
# model2.load_state_dict(state_dict_2)
# model2.eval()
# print("Acc pretrained:", ResNet18.evaluate(model2))

plt.plot(alphas, accs)
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('linear_interpolation.png')
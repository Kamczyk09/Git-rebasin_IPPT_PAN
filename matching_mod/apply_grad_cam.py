from torchvision import transforms
import pytorch_grad_cam
from PIL import Image
from data.cifar10 import load_data
import matplotlib.pyplot as plt
import models.ResNet18 as ResNet18
import torch
import numpy as np
from utils_module.utils import merge_models
import cv2

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#UWAGA ten kod robi grad-cam dla modelu ResNet18 ale dla bloku konwolucyjnego layer2 (wynika to z faktu że cifar10 ma rozdzielczość tylko 32x32)
def GradCam(model, image, res=32):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])

    input_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    image_numpy = np.array(image.resize((res, res))).astype(np.float32) / 255.0
    if image_numpy.shape[2] == 4:
        image_numpy = image_numpy[:, :, :3]

    target_layers = [model.layer2[-1]]

    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()
    targets = [pytorch_grad_cam.ClassifierOutputTarget(predicted_class)]

    with pytorch_grad_cam.LayerCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = pytorch_grad_cam.show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)

    return visualization, predicted_class


def compare_heatmaps(images_idx, models): #funkcja która wyświetla obrazy 4x3: 1 kolumna: 4 obrazy oryginalne, 2 kolumna: 4 heatmapy modelu_a, 3 kolumna: 4 heatmapy modelu_b

    images = []
    train_ds, test_ds = load_data()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for idx in images_idx:
        images.append(test_ds[idx][0])

    denormalized_images = [image * std + mean for image in images]  # tensor [3, H, W]
    np_images = [image.permute(1, 2, 0).numpy() for image in denormalized_images]

    fig, axes = plt.subplots(4, 1+len(models), figsize=(20, 20))

    for i in range(4):
        axes[i, 0].imshow(np.clip(np_images[i], 0, 1))
        # axes[i, 0].axis('off')
        image_pil = Image.fromarray((np_images[i] * 255).astype(np.uint8))

        for j, model in enumerate(models):
            visualization, predicted_class = GradCam(model=model, image=image_pil)
            axes[i, j+1].imshow(visualization)
            # axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f"Predicted class is {cifar_classes[predicted_class]}")

        if i == 0:
            axes[i, 0].set_title("No grad-cam")

    plt.savefig("grad-cam_comparsion.png")
    plt.show()

def compare_resnets(): # training two models with different initializations
    model_a = ResNet18.return_model(10)
    path_a = "models_checkpoints/resnet18_pretrained.pth"
    model_a.load_state_dict(torch.load(path_a, map_location=torch.device('cpu')))
    model_a.eval()

    model_b = ResNet18.return_model(10)
    path_b = "models_checkpoints/resnet18_pretrained_1.0.pth"
    model_b.load_state_dict(torch.load(path_b, map_location=torch.device('cpu')))
    model_b.eval()

    model_c = merge_models(model_a, model_b)
    model_c.eval()

    images_idx = [11, 110, 112, 1201]
    models = [model_a, model_b, model_c]
    compare_heatmaps(images_idx, models)

    for model in models:
        print(f"Evaluation of {model.__class__.__name__}")
        ResNet18.evaluate(model)

def compare_resnets_1(): #training one model then after some epochs dividing it into two separate models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_parent_weights = ResNet18.train(3, pretrained=True)

    model_a = ResNet18.return_model(10).to(device)
    model_a.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.00005, model=model_a, name="resnet18_T800")

    model_b = ResNet18.return_model(10).to(device)
    model_b.load_state_dict(model_parent_weights)
    ResNet18.train(10, lr=0.0001, model=model_b, name="resnet18_T1000")


    model_c = merge_models(model_a, model_b)
    model_c.eval()

    images_idx = [11, 110, 112, 1201]
    models = [model_a, model_b, model_c]
    compare_heatmaps(images_idx, models)

    for model in models:
        print(f"Evaluation of {model.__class__.__name__}")
        ResNet18.evaluate(model)

def main():
    #trening jednego modelu 5 epok. Potem rozdzielam to na dwa osobne modele i trenuję na różnych hiperparamaterach.
    # po otrzymaniu model_a i model_b wrzucam je do compare_heatmaps

    compare_resnets()

main()

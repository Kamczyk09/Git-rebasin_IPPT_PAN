import argparse
import torch
from IPython import sys_info
from torchvision import datasets, transforms, models
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vgg import VGG
from utils_module.training import test
from utils_module.weight_matching import vgg16_permutation_spec, weight_matching, apply_permutation
from utils_module.utils import  lerp
from utils_module.plot import plot_interp_acc

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_a", type=str, required=True)
#     parser.add_argument("--model_b", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=0, help="Random seed")
#     args = parser.parse_args()
#
#     # load models
#     model_a = VGG("VGG16")
#     model_b = VGG("VGG16")
#     checkpoint = torch.load(args.model_a)
#     model_a.load_state_dict(checkpoint)
#     checkpoint_b = torch.load(args.model_b)
#     model_b.load_state_dict(checkpoint_b)
#
#     permutation_spec = vgg16_permutation_spec()
#     final_permutation = weight_matching(permutation_spec,
#                                         model_a.state_dict(), model_b.state_dict())
#
#
#     updated_params = apply_permutation(permutation_spec, final_permutation, model_b.state_dict())
#
#
#     # test against mnist
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#
#
#     trainset = datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
#                                             shuffle=True, num_workers=2)
#
#     testset = datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
#                                             shuffle=False, num_workers=2)
#
#     lambdas = torch.linspace(0, 1, steps=25)
#
#     test_acc_interp_clever = []
#     test_acc_interp_naive = []
#     train_acc_interp_clever = []
#     train_acc_interp_naive = []
#     # naive
#     model_b.load_state_dict(checkpoint_b)
#     model_a_dict = copy.deepcopy(model_a.state_dict())
#     model_b_dict = copy.deepcopy(model_b.state_dict())
#     for lam in tqdm(lambdas):
#       naive_p = lerp(lam, model_a_dict, model_b_dict)
#       model_b.load_state_dict(naive_p)
#       test_loss, acc = test(model_b.cuda(), 'cuda', test_loader, True)
#       test_acc_interp_naive.append(acc)
#       train_loss, acc = test(model_b.cuda(), 'cuda', train_loader, True)
#       train_acc_interp_naive.append(acc)
#
#     # smart
#     model_b.load_state_dict(updated_params)
#     model_b.cuda()
#     model_a.cuda()
#     model_a_dict = copy.deepcopy(model_a.state_dict())
#     model_b_dict = copy.deepcopy(model_b.state_dict())
#     for lam in tqdm(lambdas):
#       naive_p = lerp(lam, model_a_dict, model_b_dict)
#       model_b.load_state_dict(naive_p)
#       test_loss, acc = test(model_b.cuda(), 'cuda', test_loader, True)
#       test_acc_interp_clever.append(acc)
#       train_loss, acc = test(model_b.cuda(), 'cuda', train_loader, True)
#       train_acc_interp_clever.append(acc)
#
#     fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
#                     train_acc_interp_clever, test_acc_interp_clever)
#     plt.savefig(f"cifar10_vgg_weight_matching_interp_accuracy_epoch.png", dpi=300)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    model_a = VGG("VGG16").to(device)
    model_b = VGG("VGG16").to(device)
    checkpoint = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(checkpoint)
    checkpoint_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(checkpoint_b)

    permutation_spec = vgg16_permutation_spec()
    final_permutation = weight_matching(permutation_spec,
                                        model_a.state_dict(), model_b.state_dict())

    updated_params = apply_permutation(permutation_spec, final_permutation, model_b.state_dict())

    # CIFAR10 dataset & loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                              shuffle=False, num_workers=2)

    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []

    # naive interpolation
    model_b.load_state_dict(checkpoint_b)
    model_b.to(device)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(naive_p)
        test_loss, acc = test(model_b, device, test_loader, True)
        test_acc_interp_naive.append(acc)
        train_loss, acc = test(model_b, device, train_loader, True)
        train_acc_interp_naive.append(acc)

    # smart interpolation
    model_b.load_state_dict(updated_params)
    model_b.to(device)
    model_a.to(device)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(naive_p)
        test_loss, acc = test(model_b, device, test_loader, True)
        test_acc_interp_clever.append(acc)
        train_loss, acc = test(model_b, device, train_loader, True)
        train_acc_interp_clever.append(acc)

    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                         train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig("cifar10_vgg_weight_matching_interp_accuracy_epoch.png", dpi=300)


if __name__ == "__main__":
  main()
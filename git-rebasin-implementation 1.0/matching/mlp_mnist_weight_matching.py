import argparse
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mlp import MLP
from utils_module.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils_module.utils import flatten_params, lerp
from utils_module.plot import plot_interp_acc
from utils_module.training import test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    model_a = MLP().to(device)
    model_b = MLP().to(device)
    checkpoint = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(checkpoint)
    checkpoint_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(checkpoint_b)

    #TWORZENIE PRZESTRZENI WAG PO PERMUTACJI
    permutation_spec = mlp_permutation_spec(4) #tworzy mapę - jakie permutacje trzeba zastosować dla konkretnych warstw i osi (weight lub bias)

    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
    
    for key in final_permutation:
        print(key)
        print(final_permutation[key].shape)

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))
    ########################################


    # test against mnist
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_kwargs = {'batch_size': 5000}
    train_kwargs = {'batch_size': 5000}
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []

    # naive
    model_b.load_state_dict(checkpoint_b)
    model_b.to(device)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(naive_p)
        test_loss, acc = test(model_b.to(device), device, test_loader)
        test_acc_interp_naive.append(acc)
        train_loss, acc = test(model_b.to(device), device, train_loader)
        train_acc_interp_naive.append(acc)

    # smart
    model_b.load_state_dict(updated_params)
    model_b.to(device)
    model_a.to(device)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
        naive_p = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(naive_p)
        test_loss, acc = test(model_b.to(device), device, test_loader)
        test_acc_interp_clever.append(acc)
        train_loss, acc = test(model_b.to(device), device, train_loader)
        train_acc_interp_clever.append(acc)

    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                          train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig("mnist_mlp_weight_matching_interp_accuracy_epoch.png", dpi=300)

if __name__ == "__main__":
  main()
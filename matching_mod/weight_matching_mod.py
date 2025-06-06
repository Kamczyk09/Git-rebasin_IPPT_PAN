import argparse
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

import ResNet18 as model_func
from utils_module.training import test
from utils_module.weight_matching import weight_matching, apply_permutation, resnet18_permutation_spec
from utils_module.utils import  lerp
from utils_module.plot import plot_interp_acc
from data.medical_images import load_data
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Checkpoint 0")

    ######LOADING DATA######
    trainset, testset = load_data(pretrained=True) #!!!!! true czy false
    ########################
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(trainset, batch_size=1024, shuffle=False)
    num_classes = len(trainset.classes)

    # load models
    model_a = model_func.return_model(nOutputNeurons=num_classes)
    checkpoint = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(checkpoint)
    model_b = model_func.return_model(nOutputNeurons=num_classes)
    checkpoint_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(checkpoint_b)

    permutation_spec = resnet18_permutation_spec() # stworzyć taką funkcję w utils_module.weight_matching

    final_permutation = weight_matching(permutation_spec,
                                        model_a.state_dict(), model_b.state_dict())


    updated_params = apply_permutation(permutation_spec, final_permutation, model_b.state_dict())
    #

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
    plt.savefig(f"{args.seed}_weight_matching_interp_accuracy_epoch.png")


if __name__ == "__main__":
  main()
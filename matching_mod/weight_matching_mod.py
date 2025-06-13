import argparse
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from utils_module.training import test
from utils_module.weight_matching import weight_matching, apply_permutation, resnet18_permutation_spec, cnn_permutation_spec, mlp_permutation_spec
from utils_module.utils import  lerp
from utils_module.plot import plot_interp_acc
from data.cifar10 import load_data
from torch.utils.data import DataLoader

model_type = 'mlp'

if model_type == 'cnn':
    import CNN as model_func
elif model_type == 'resnet18':
    import ResNet18 as model_func
elif model_type == 'mlp':
    import MLP as model_func
else:
    raise("Wrong model_type")


# /home/skaminsk/Pulpit/matching_mod/models_checkpoints/resnet18_pretrained.pth
# /home/skaminsk/Pulpit/matching_mod/models_checkpoints/resnet18_raw.pth
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, default=None)
    parser.add_argument("--model_b", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.model_a is None:
        args.model_a = f'models_checkpoints/{model_type}_raw.pth'
        args.model_b = f'models_checkpoints/{model_type}_pretrained.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Checkpoint 0")

    ######LOADING DATA######
    trainset, testset = load_data()
    ########################
    train_loader = DataLoader(trainset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)
    num_classes = 1 if len(trainset.classes)==2 else len(trainset.classes)

    print("Checkpoint 1 - loading data")
    print(num_classes)

    # load models
    model_a = model_func.return_model(nOutputNeurons=num_classes).to(device)
    checkpoint = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(checkpoint)
    model_b = model_func.return_model(nOutputNeurons=num_classes).to(device)
    checkpoint_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(checkpoint_b)

    if model_type=="cnn":
        permutation_spec = cnn_permutation_spec()
    elif model_type=="resnet18":
        permutation_spec = resnet18_permutation_spec()
    elif model_type=="mlp":
        permutation_spec = mlp_permutation_spec(4)
    else:
        raise("Wrong model_type")

    state_dict_a = {k: v.to(device) for k, v in model_a.state_dict().items()}
    state_dict_b = {k: v.to(device) for k, v in model_b.state_dict().items()}

    final_permutation = weight_matching(permutation_spec,
                                        state_dict_a, state_dict_b)

    updated_params = apply_permutation(permutation_spec,
                                       final_permutation, state_dict_b)

    print("Checkpoint 2 - loading models")

    lambdas = torch.linspace(0, 1, steps=30)

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

    print("train_acc_interp_naive:", train_acc_interp_naive)
    print("train_acc_interp_clever:", train_acc_interp_clever)
    print("test_acc_interp_naive:", test_acc_interp_naive)
    print("test_acc_interp_clever:", test_acc_interp_clever)
    print("lambdas:", lambdas)

    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                         train_acc_interp_clever, test_acc_interp_clever)
    plt.savefig(f"{args.seed}_weight_matching_{model_type}.png")


if __name__ == "__main__":
  main()
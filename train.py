"""
Train and save best models parameters
Attention: this script always will rewrite model

Command usage: `python train.py [options]`
Example `python train.py --epochs=10`
Options syntax: --epochs=<value>
Options:
    1) --epochs - number of epochs for training
    2) --kernel_size - choose pane
    3) --model - model achitecture (resnet20, resnet32, resnet44)
    4) --filename - name of the folder, containing model parameters and configs
"""
import argparse

import numpy as np

import torch
import torchvision
from torchsummary import summary

from resnet import resnet20, resnet32, resnet44
from utils import train_model, test_model, save_model


# Main function ================================================================================
def run(
        epochs: int = 10,
        seed: int = 42,
        filename: str = None,
        model_name: str = None) -> None:
    # Set seed for random generators for reproducability ------------------------------------------------#
    SEED = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # Number of epochs ------------------------------------------------------------#
    NUM_EPOCHS = epochs

    # Batch size ------------------------------------------------------------#
    BATCH_SIZE = 128

    # Learning rate ------------------------------------------------------------#
    LEARNING_RATE = 1e-3

    # Enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)

    # Data loaders -----------------------------------------------------------------#
    train_dataset = torchvision.datasets.EMNIST(
        root="D:\\university_materials\\research_project\\Lowbit_NN_Quantization\\data\\",
        split="mnist",
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    test_dataset = torchvision.datasets.EMNIST(
        root="D:\\university_materials\\research_project\\Lowbit_NN_Quantization\\data\\",
        split="mnist",
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model selection -------------------------------------------------------------#
    if model_name.lower() == "resnet20":
        model = resnet20().to(device)
    elif model_name.lower() == "resnet32":
        model = resnet32().to(device)
    elif model_name.lower() == "resnet44":
        model = resnet44().to(device)
    else:
        raise ValueError(
            "Unvalid model achitecture, choose from resnet20, resnet32, resnet44")

    summary(model, (1, 28, 28))

    # Hyperparameter selection ----------------------------------------------------#
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Parameters for stopping -----------------------------------------------------#
    threshold = 0.9955

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[20, 30, 50],
        verbose=True
    )

    # Training process ------------------------------------------------------------#
    history = train_model(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=NUM_EPOCHS,
        device=device,
        stop_after_test_threshold=True,
        threshold=threshold,
        test_dataloader=test_dataloader,
        seed=SEED
    )

    # Resulting accuracy
    print("Accuracy")
    test_model(
        model=model,
        dataloader=test_dataloader,
        qat=False
    )

    # Model saving ===========================================
    save_model(
        model=model,
        model_name=filename,
        info={
            "seed": seed,
            "model_achitecture": model_name,
            "epochs": NUM_EPOCHS,
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--model", default="resnet20", type=str)

    parser.add_argument("--filename", default="model_from_train", type=str)

    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    print("Training settings: ")
    print(args)

    run(
        epochs=args.epochs,
        seed=args.seed,
        filename=args.filename,
        model_name=args.model,
    )

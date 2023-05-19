import os
import sys
import time
import json
from datetime import datetime

import torch

from resnet import resnet20, resnet32, resnet44


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        res_time = (time.time() - start_time) / 60
        print(
            f"Time consumption of func {func.__name__} = {res_time:.2f} mins")

    return wrapper


@time_it
def train_model(
        model: torch.nn.Module,
        dataloader,
        loss_fn,
        optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs,
        device='cpu',
        qat: bool = False,
        bit_width: int = 8,
        stop_after_test_threshold: bool = False,
        test_dataloader: torch.utils.data.DataLoader = None,
        threshold: float = 0.9955,
        seed: int = None) -> list:
    """
    Implements training loop for model
    Returns history of loss and accuracy during epochs
    """
    if qat:
        print(f"Bitwidth for QAT: {bit_width}")

    history = {'loss': [], 'accuracy': []}

    steps_per_epoch = len(dataloader)
    model.train()
    model.to(device)

    n_total = len(dataloader.dataset)

    checkpoint_version = 1

    for epoch in range(epochs):
        cum_loss = 0
        n_correct = 0
        start_time = time.time()

        for (features, labels) in dataloader:
            # Move batches to device (cuda if available)
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            # apply here crossentropy, so you don't need to use softmax
            loss = loss_fn(logits, labels)
            cum_loss += loss.item()

            # Check accuracy on train dataset
            # Here you need softmax for accuracy tracking
            predictions = torch.log_softmax(logits, dim=1)
            predictions = torch.argmax(input=predictions, dim=1)
            n_correct += (labels == predictions).sum().item()

            # Calculate gradient and make gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(
            epoch=epoch
        )

        # Save loss and accuracy
        history['loss'].append(cum_loss / steps_per_epoch)
        history['accuracy'].append(n_correct / n_total)

        # Loss tracking between different epochs
        print(f"Epoch {epoch+1}")
        print("-------------------------------")
        print(f"Average loss = {cum_loss / steps_per_epoch:.5f}")

        # Accuracy tracking
        print(f"Average accuracy = {n_correct / n_total:.5f}")

        # Accuracy on test dataset
        if test_dataloader != None:
            print("Accuracy on test dataset")
            test_acc = test_model(
                model=model,
                dataloader=test_dataloader,
                qat=qat,
                bitwidth=bit_width,
                device=device
            )
            print(test_acc)

            if stop_after_test_threshold:
                if test_acc >= threshold:
                    print(f"Model reached needed threshold: {threshold}")
                    save_model(
                        model=model,
                        model_name=f"BestModel_V{checkpoint_version}",
                        info={
                            "seed": seed,
                            "model_achitecture": model._get_name(),
                            "epochs": epoch+1
                        }
                    )
                    checkpoint_version += 1

        # Time tracking
        res_time = time.time() - start_time

        print(f"Time for this epoch = {res_time:.2f} secs")
        print("or in minutes")
        res_time_mins = res_time / 60
        print(f"Time for this epoch = {res_time_mins:.2f} mins")

        start_date = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        print(f"Start time: {start_date}")

        end_date = datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
        print(f"End time: {end_date}")

        print("-------------------------------")


# @time_it
def test_model(
        model: torch.nn.Module,
        dataloader,
        qat: bool = False,
        bitwidth: int = 8,
        device: str = 'cuda') -> None:
    """
    This function implements accuracy measuring on passed dataloader on cuda
    """
    model.to(device)
    model.eval()

    n_correct = 0
    n_total = len(dataloader.dataset)

    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = torch.log_softmax(logits, dim=1)
            predictions = torch.argmax(input=predictions, dim=1)

            try:
                n_correct += (labels == predictions).sum().item()
            except RuntimeError:
                print(f"Shape of images: {images.shape}")
                print(f"Shape of logits: {logits.shape}")
                print(f"Shape of predictions: {predictions.shape}")
                print(f"Shape of labels: {labels.shape}")
                sys.exit(-1)

    print(f"Accuracy = {n_correct / n_total*100:.2f}%")

    return n_correct / n_total


def save_model(model: torch.nn.Module, model_name: str, info: dict):
    if os.path.exists(path="checkpoints"):
        if os.path.exists(path=f"checkpoints/{model_name}/{model_name}.pt"):
            print("Rewrite old model data")
        else:
            os.mkdir(f"checkpoints/{model_name}")
        torch.save(
            obj=model.state_dict(),
            f=f"checkpoints/{model_name}/{model_name}.pt"
        )

        with open(f"checkpoints/{model_name}/{model_name}.info.json", "w") as file:
            json.dump(
                obj=info,
                fp=file
            )
    else:
        raise FileNotFoundError("Not found folder checkpoints")


def load_model(model_name: str) -> torch.nn.Module:
    """
    Filename - file, containing state dict from saved model (without folder name and extension)
    """

    with open(f"checkpoints/{model_name}/{model_name}.info.json", "r") as config_file:
        configs = json.load(config_file)

    model_achitecture = configs['model_achitecture']
    if model_achitecture.lower() == "resnet20":
        model = resnet20()
    elif model_achitecture.lower() == "resnet32":
        model = resnet32()
    elif model_achitecture.lower() == "resnet44":
        model = resnet44()
    else:
        raise ValueError("Something gone wrong with load model, model_achitecture not valid")

    # model.load_state_dict(torch.load(filename))
    model.load_state_dict(torch.load(
        f"checkpoints/{model_name}/{model_name}.pt"))
    return model

from pathlib import Path

import torch
from torchvision import datasets, transforms


PATH_ROOT = Path(".")


def main():

    cifar_path = PATH_ROOT.joinpath("CIFAR10")
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    val_and_trainset = datasets.CIFAR10(
        root=cifar_path,
        train=True,
        transform=train_transforms,
        download=True,
    )

    testset = datasets.CIFAR10(
        root=cifar_path,
        train=False,
        transform=test_transforms,
        download=True,
    )

    dataset_seed = 42
    trainset, valset = torch.utils.data.random_split(
        val_and_trainset,
        [40000, 10000],
        generator=torch.Generator().manual_seed(dataset_seed),
    )

    # save dataset and seed in data directory
    dataset = {
        "trainset": trainset,
        "valset": valset,
        "testset": testset,
        "dataset_seed": dataset_seed,
    }
    torch.save(dataset, cifar_path.joinpath("dataset.pt"))

if __name__ == "__main__":
    main()

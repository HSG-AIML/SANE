from pathlib import Path

import ray
from ray import tune

import torch
from torchvision import datasets, transforms

# load datastet
from tiny_imagenet_helpers import TinyImageNetDataset

PATH_ROOT = Path(".")


def main():

    data_path = PATH_ROOT.joinpath("tiny-imagenet-200")

    # normalization computed with:
    # https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
    train_transforms = transforms.Compose(
        [
    #         transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
    #         transforms.ToTensor(),
            transforms.Normalize(
            mean=[255*0.485, 255*0.456, 255*0.406],
            std=[255*0.229, 255*0.224, 255*0.225],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
    #         transforms.ToTensor(),
            transforms.Normalize(
            mean=[255*0.485, 255*0.456, 255*0.406],
            std=[255*0.229, 255*0.224, 255*0.225],
            ),
        ]
    )

    trainset = TinyImageNetDataset(
        root_dir=data_path,
        mode='train',
        preload=True,
        load_transform=None,
        transform=train_transforms,
        download=False,
        max_samples=None
        )

    testset = TinyImageNetDataset(
        root_dir=data_path,
        mode='val',
        preload=True,
        load_transform=None,
        transform=test_transforms,
        download=False,
        max_samples=None
        )

    # save dataset and seed in data directory
    dataset = {
        "trainset": trainset,
        "testset": testset,
    }
    torch.save(dataset, data_path.joinpath("dataset.pt"))



if __name__ == "__main__":
    main()

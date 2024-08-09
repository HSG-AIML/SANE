import os
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PreprocessedSamplingDataset(Dataset):
    def __init__(self, zoo_paths, split="train", transforms=None):
        self.zoo_paths = zoo_paths
        self.split = split
        self.datasets = self.load_datasets(zoo_paths)
        self.transforms = transforms
        self.samples = self.collect_samples(self.datasets)

    def load_datasets(self, zoo_paths):
        datasets = []
        for path in zoo_paths:
            directory_path = (
                Path(path).joinpath(f"dataset_torch.{self.split}").absolute()
            )
            if os.path.isdir(directory_path):
                datasets.append(directory_path)
            else:
                raise NotADirectoryError(f"Directory not found: {directory_path}")
        return datasets

    def collect_samples(self, datasets):
        samples = []
        for dataset in datasets:
            for file_name in os.listdir(dataset):
                file_path = os.path.join(dataset, file_name)
                if os.path.isfile(file_path):
                    samples.append(file_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        item = torch.load(file_path)

        ddx = item["w"]
        mask = item["m"]
        pos = item["p"]
        props = item["props"]

        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)
        return ddx, mask, pos, props

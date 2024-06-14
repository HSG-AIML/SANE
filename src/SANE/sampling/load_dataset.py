import torch
from SANE.models.def_net import NNmodule
import logging
import copy
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader


def load_datasets_from_config(config):
    if config.get("dataset::dump", None) is not None:
        print(f"loading data from {config['dataset::dump']}")
        dataset = torch.load(config["dataset::dump"])
        trainset = dataset["trainset"]
        testset = dataset["testset"]
        valset = dataset.get("valset", None)
    else:
        data_path = config["training::data_path"]
        fname = f"{data_path}/train_data.pt"
        train_data = torch.load(fname)
        train_data = torch.stack(train_data)
        fname = f"{data_path}/train_labels.pt"
        train_labels = torch.load(fname)
        train_labels = torch.tensor(train_labels)
        # test
        fname = f"{data_path}/test_data.pt"
        test_data = torch.load(fname)
        test_data = torch.stack(test_data)
        fname = f"{data_path}/test_labels.pt"
        test_labels = torch.load(fname)
        test_labels = torch.tensor(test_labels)
        #
        # Flatten images for MLP
        if config["model::type"] == "MLP":
            train_data = train_data.flatten(start_dim=1)
            test_data = test_data.flatten(start_dim=1)
        # send data to device
        # if config["cuda"]:
        #     train_data, train_labels = train_data.cuda(), train_labels.cuda()
        #     test_data, test_labels = test_data.cuda(), test_labels.cuda()
        # else:
        #     print(
        #         "### WARNING ### : using tensor dataloader without cuda. probably slow"
        #     )
        # create new tensor datasets
        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # instanciate Tensordatasets
    dl_type = config.get("training::dataloader", "tensor")
    if dl_type == "tensor":
        trainloader = FastTensorDataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            # num_workers=config.get("testloader::workers", 2),
        )
        testloader = FastTensorDataLoader(
            dataset=testset, batch_size=len(testset), shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = FastTensorDataLoader(
                dataset=valset, batch_size=len(valset), shuffle=False
            )

    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            num_workers=config.get("testloader::workers", 2),
        )
        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=config["training::batchsize"], shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = torch.utils.data.DataLoader(
                dataset=valset, batch_size=config["training::batchsize"], shuffle=False
            )

    return trainloader, testloader, valloader

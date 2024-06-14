import torch
from SANE.models.def_net import NNmodule
import logging
import copy
from collections import OrderedDict
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from SANE.sampling.load_dataset import load_datasets_from_config


def evaluate_ensemble(
    config: dict, checkpoints: OrderedDict, fintuning_epochs: int = 0
) -> dict:
    """
    evaluates a single model on a single task
    Args:
        config (dict): dictionary containing config for the model
        checkpoint (OrderedDict): state dict of the model
        fintuning_epochs (int): number of epochs to finetune
    Returns:
        dict: dictionary containing evaluation results
    """
    # init output
    results = {}
    # load datasets
    trainloader, testloader, valloader = load_datasets_from_config(config)
    # set parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = fintuning_epochs if fintuning_epochs != 0 else 1
    config["scheduler::steps_per_epoch"] = len(trainloader)
    # iterate over models
    y_train = []
    y_test = []
    y_val = []
    t_train = []
    t_test = []
    t_val = []
    for cdx, checkpoint in enumerate(checkpoints):
        # init model
        logging.info("initialize sampled model")
        module = NNmodule(config, cuda=True, verbosity=0)
        # load checkpoint
        logging.info("load checkpoint model")
        module.model.load_state_dict(checkpoint)
        # send model to device
        """
        compilation currently throws cryptic error, so leaving this out for now
        # compile model
        print(f"attempt model compilation")
        # cuda before compile :) https://discuss.pytorch.org/t/torch-compile-before-or-after-cuda/176031
        module.compile_model()
        print(f"model compiled...")
        """
        # finetune model
        for idx in range(fintuning_epochs):
            loss_train, acc_train = module.train_epoch(trainloader, 0)

        # get predictions for train, test val set
        y_train_tmp = []
        y_test_tmp = []
        y_val_tmp = []
        with torch.no_grad():
            for idx, (x, t) in enumerate(trainloader):
                x = x.to(module.device)
                y_train_tmp.append(module.model(x).detach().cpu())
                if cdx == 0:
                    t_train.append(t)
            for idx, (x, t) in enumerate(testloader):
                x = x.to(module.device)
                y_test_tmp.append(module.model(x).detach().cpu())
                if cdx == 0:
                    t_test.append(t)
            if valloader is not None:
                for idx, (x, t) in enumerate(valloader):
                    x = x.to(module.device)
                    y_val_tmp.append(module.model(x).detach().cpu())
                    if cdx == 0:
                        t_val.append(t)
        # stack predictions
        y_train_tmp = torch.cat(y_train_tmp, dim=0)
        y_test_tmp = torch.cat(y_test_tmp, dim=0)
        if valloader is not None:
            y_val_tmp = torch.cat(y_val_tmp, dim=0)

        # stack targets
        if cdx == 0:
            t_train = torch.cat(t_train, dim=0)
            t_test = torch.cat(t_test, dim=0)
            if valloader is not None:
                t_val = torch.cat(t_val, dim=0)

        # append to list
        y_train.append(y_train_tmp)
        y_test.append(y_test_tmp)
        if valloader is not None:
            y_val.append(y_val_tmp)

        logging.debug(f"model {cdx} y_train_tmp.shape: {y_train_tmp.shape}")
        logging.debug(f"model {cdx} y_test_tmp.shape: {y_test_tmp.shape}")

    # stack predictions
    y_train = torch.stack(y_train)
    y_test = torch.stack(y_test)
    if valloader is not None:
        y_val = torch.stack(y_val)
    logging.debug(f"y_train.shape: {y_train.shape}")
    logging.debug(f"y_test.shape: {y_test.shape}")
    # average predictions
    y_train = y_train.mean(dim=0)
    y_test = y_test.mean(dim=0)
    if valloader is not None:
        y_val = y_val.mean(dim=0)
    # compute accuracy
    acc_train = (y_train.argmax(dim=1) == t_train).sum().item() / t_train.shape[0]
    acc_test = (y_test.argmax(dim=1) == t_test).sum().item() / t_test.shape[0]
    results["acc_test"] = [acc_test]
    results["acc_train"] = [acc_train]
    if valloader is not None:
        acc_val = (y_val.argmax(dim=1) == t_val).sum().item() / t_val.shape[0]
        results["acc_val"] = [acc_val]
    return results

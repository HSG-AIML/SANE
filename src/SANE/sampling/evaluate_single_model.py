import torch
from SANE.models.def_net import NNmodule
import logging
import copy
from collections import OrderedDict
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from SANE.sampling.load_dataset import load_datasets_from_config


def evaluate_single_model(
    config: dict, checkpoint: OrderedDict, fintuning_epochs: int = 0
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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"try to put model to {device}")
    # assert device == "cuda", "device is not cuda, fine-tuning is going to take forever"

    # # send model to device
    # logging.info(f"device: {device}")
    # module.model.to(device)

    # eval zero shot
    loss_train, acc_train = module.test_epoch(trainloader, 0)
    loss_test, acc_test = module.test_epoch(testloader, 0)
    results["loss_train"] = [loss_train]
    results["acc_train"] = [acc_train]
    results["loss_test"] = [loss_test]
    results["acc_test"] = [acc_test]
    if valloader is not None:
        loss_val, acc_val = module.test_epoch(valloader, 0)
        results["loss_val"] = [loss_val]
        results["acc_val"] = [acc_val]
    # finetune model
    for idx in range(fintuning_epochs):
        loss_train, acc_train = module.train_epoch(trainloader, 0)
        loss_test, acc_test = module.test_epoch(testloader, 0)
        results["loss_train"].append(loss_train)
        results["acc_train"].append(acc_train)
        results["loss_test"].append(loss_test)
        results["acc_test"].append(acc_test)
        if valloader is not None:
            loss_val, acc_val = module.test_epoch(valloader, 0)
            results["loss_val"].append(loss_val)
            results["acc_val"].append(acc_val)
    # return results
    return results

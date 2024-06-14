from collections import OrderedDict
from SANE.datasets.dataset_auxiliaries import tokens_to_checkpoint, tokenize_checkpoint
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from SANE.models.def_NN_experiment import NNmodule

import torch

from typing import Optional, List, Any

import numpy as np

from einops import repeat

from sklearn.neighbors import KernelDensity

import logging


##
def sample_model_evaluation(
    sample_config: dict,
    finetuning_epochs: int,
    repetitions: int,
    anchor_ds_path: str,
    norm_mode: Optional[str] = None,
    layer_norms: Optional[dict] = None,
    reset_classifier: bool = False,
) -> dict:
    """
    runs evaluation pipeline.
    randomly draws #repetitions models from anchor_ds and finetunes checkpoints on downstream task and evaluates finetuned checkpoints
    Args:
        sample_config (dict): dictionary containing config for sampled model
        finetuning_epochs (int): number of epochs to finetune
        repetitions (int): number of repetitions to finetune and evaluate
    Returns:
        dict: dictionary containing evaluation results
    """
    # init output
    results = {}
    # get reference model tokens
    logging.info("sampling:: create reference checkoint")
    sample_config["scheduler::steps_per_epoch"] = 123
    module = NNmodule(sample_config)
    checkpoint_ref = module.model.state_dict()

    # sample models
    logging.info("sampling:: sample models")
    checkpoints = sample_models(
        checkpoint_ref=checkpoint_ref,
        anchor_ds_path=anchor_ds_path,
        repetitions=repetitions,
        reset_classifier=reset_classifier,
    )

    # de-normalize checkoints
    logging.info("sampling:: de-normalize checkpoints")
    if norm_mode is not None:
        for idx in range(len(checkpoints)):
            checkpoints[idx] = de_normalize_checkpoint(
                checkpoints[idx], layers=layer_norms, mode=norm_mode
            )

    # evaluate models
    logging.info("sampling:: evaluate models")
    for rep in range(repetitions):
        # sample model
        checkpoint = checkpoints[rep]
        # finetune and evaluate model
        res_tmp = evaluate_single_model(sample_config, checkpoint, finetuning_epochs)
        # append results
        for k in res_tmp.keys():
            results[f"model_{rep}_{k}"] = res_tmp[k]

    # aggregate results over models
    for k in res_tmp.keys():
        res_tmp = []
        for rep in range(repetitions):
            res_tmp.append(results[f"model_{rep}_{k}"])

        results[f"{k}_mean"] = []
        results[f"{k}_std"] = []
        for idx in range(len(res_tmp[0])):
            res_ep = [res_tmp[jdx][idx] for jdx in range(len(res_tmp))]
            results[f"{k}_mean"].append(np.mean(res_ep))
            results[f"{k}_std"].append(np.std(res_ep))

    # return results
    return results


def sample_models(
    checkpoint_ref: OrderedDict,
    repetitions: int,
    anchor_ds_path: str,
    reset_classifier: bool = False,
) -> OrderedDict:
    """
    Sample a single model from a hyper-representation model.
    Args:
        checkpoint_ref (str): Reference checkpoint for the sampled model
        pos (int): reference positions for sampling of shape [windowsize,3]
        mask (torch.Tensor): reference mask for sampling
        repetitions (int): number of repetitions to sample
    Returns:
        List[OrderedDict]: sampled model state dicts
    """
    # call the hyper-representation model's sample method
    # tkdx = torch.randn(mask.shape)

    # get anchor model weights, pos
    anchor_ds = torch.load(anchor_ds_path)
    anchor_weights, anchor_masks = anchor_ds.__get_weights__()
    anchor_w_shape = anchor_weights.shape

    # tokenize reference checkpoint to check equivalence
    tok_ref, mask_ref, pos_ref = tokenize_checkpoint(
        checkpoint=checkpoint_ref,
        tokensize=anchor_ds.tokensize,
        return_mask=True,
        ignore_bn=False,
    )

    # draw random samples from anchor dataset
    sampled_ids = torch.randperm(
        n=anchor_weights.shape[-3], dtype=torch.int32, device="cpu"
    )[:repetitions]
    # slice
    sampled_tokens = torch.index_select(anchor_weights, -3, sampled_ids).squeeze()
    sampled_tokens = sampled_tokens.detach().cpu().to(torch.float)
    assert (
        sampled_tokens.shape[0] == repetitions
    ), f"first dimension (sample size) of sampled tokens {sampled_tokens.shape} needs to match repetitions {repetitions}"
    assert (
        sampled_tokens.shape[1:] == anchor_w_shape[1:]
    ), f"sequence and token dimensions of sampled tokens {sampled_tokens.shape} and tmp anchor tokens {anchor_w_shape} need to match"

    logging.debug(f"generated tokens:{sampled_tokens.shape}")

    # if reference and source sequences are not the same, pad last tokens with zeros
    # we assume difference are only in the last layer at this point
    if not sampled_tokens.shape[1] == tok_ref.shape[0]:
        if not reset_classifier:
            raise ValueError(
                f"sampled tokens {sampled_tokens.shape} and reference tokens {tok_ref.shape} differ in sequence length. set reset_classifier=True to re-initialize classifier head"
            )
        # pad last tokens with zeros
        # sampled_tokens has shape [n_reps, n_tokens-source, tokendim]
        # z_ref has shape [n_tokens-ref, tokendim]
        # z_tmp has shape [n_reps, n_tokens-ref, tokendim]
        # case 1: new model is bigger
        logging.info(
            f'f"sampled tokens {sampled_tokens.shape} and reference tokens {tok_ref.shape} '
        )
        if tok_ref.shape[0] > sampled_tokens.shape[1]:
            tmp = torch.zeros(
                [sampled_tokens.shape[0], tok_ref.shape[0], sampled_tokens.shape[2]]
            )
            tmp[:, : sampled_tokens.shape[1], :] = sampled_tokens
            sampled_tokens = tmp
        # case 2: new model is smaller
        else:
            sampled_tokens = sampled_tokens[:, : tok_ref.shape[0], :]

    logging.debug(f"zero-padded generated tokens:{sampled_tokens.shape}")

    # create new state dicts
    returns = []
    for idx in range(repetitions):
        checkpoint = tokens_to_checkpoint(
            sampled_tokens[idx], pos_ref.squeeze(), checkpoint_ref, ignore_bn=False
        )
        returns.append(checkpoint)
    logging.debug(f"generated {len(returns)} checkpoints")

    # re-initialize classifier head from scratch if necessary
    if reset_classifier:
        logging.info("re-initialize classifier head")
        # infer last layer key
        layer_key = None
        ignore_bn = False
        for key in checkpoint_ref.keys():
            if "weight" in key:
                # get correct slice of modules out of vec sequence
                if ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue
                layer_key = key

        # get dimensions for last layer
        out_c = checkpoint_ref[layer_key].shape[0]
        in_c = checkpoint_ref[layer_key].shape[1]

        for idx, check in enumerate(returns):
            # re-initialize last layer
            tmp_layer = torch.nn.Linear(
                in_features=in_c, out_features=out_c, bias=True, device="cpu"
            )
            # copy weights
            check[layer_key] = tmp_layer.weight.data
            # copy biases
            if layer_key.replace("weight", "bias") in checkpoint.keys():
                check[layer_key.replace("weight", "bias")] = tmp_layer.bias.data
            # hard copy returns to make sure it's overwritten..
            returns[idx] = check

        logging.debug(f"re-initialized classifiers for {len(returns)} checkpoints")

    return returns


def de_normalize_checkpoint(checkpoint, layers, mode="minmax"):
    """
    revert normalization
    """
    # iterate over layer keys instead of checkpoint keys
    # that way, we only consider layers for which we have norm values
    for key in layers.keys():
        if key == "mode":
            continue
        if mode == "standardize":
            # get mean and std
            mu = layers[key]["mean"]
            sigma = layers[key]["std"]
            # de-normalize weights
            checkpoint[key] = checkpoint[key] * sigma + mu
            # de-noramlize bias
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] * sigma + mu
                )
        elif mode == "minmax":
            # get global min and max values
            min_glob = layers[key]["min"]
            max_glob = layers[key]["max"]
            # reverse of min-max normalization (mapped to range [-1,1])
            # returns weights exactly to original range
            checkpoint[key] = (checkpoint[key] + 1) * (
                max_glob - min_glob
            ) / 2 + min_glob
            # de-normalize bais
            if key.replace("weight", "bias") in checkpoint:
                checkpoint[key.replace("weight", "bias")] = (
                    checkpoint[key.replace("weight", "bias")] + 1
                ) * (max_glob - min_glob) / 2 + min_glob

    return checkpoint


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
    config["training::epochs_train"] = fintuning_epochs
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

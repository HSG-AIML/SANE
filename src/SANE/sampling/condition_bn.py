import torch
from SANE.models.def_net import NNmodule
import logging
import copy
from SANE.datasets.def_FastTensorDataLoader import FastTensorDataLoader


def condition_bn(config, checkpoint, dataloader, iterations=10):
    """
    generated resnets with batchnorm layers struggle with running_mean / running_var values, which often produce nan values
    as a fix, we perform #iterations __forward__ passes through the model to condition the batchnorm statistics
    no gradients are computed, no weights updated.
    """
    #
    logging.info("initialize model for conditioning")
    resnet = NNmodule(config, cuda=True, verbosity=0)
    resnet.model.load_state_dict(checkpoint)
    device = resnet.device
    # set model to train mode (s.t. bn stats can be adjusted
    resnet.train()
    for idx, batch in enumerate(dataloader):
        # check stopping criterion
        if idx == iterations:
            break
        (imgx, _) = batch
        imgx = imgx.to(device)
        with torch.no_grad():
            _ = resnet.forward(imgx)
    # set model back to eval mode
    resnet.eval()
    # send model to cpu to get cpu state_dict
    resnet.model.to("cpu")
    # get state_dict
    state_out = resnet.model.state_dict()
    # return model
    return state_out


def condition_checkpoints(checkpoints, config, iterations=10):
    # load datasets
    config["training::batchsize"] = 32
    trainloader, testloader, valloader = load_datasets_from_config(config)
    if valloader is not None:
        dataloader = valloader
    else:
        dataloader = trainloader
    # set parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = 123
    config["scheduler::steps_per_epoch"] = 123
    # init model
    # load checkpoint]
    logging.info(
        f"monitoring: same checkpoints just within bn_conditioning: {check_equivalence(checkpoints[0],checkpoints[-1])}"
    )
    check_out = []
    for idx, _ in enumerate(checkpoints):
        check = copy.deepcopy(checkpoints[idx])
        # condition
        check_new = condition_bn(config, check, dataloader, iterations=iterations)
        # replace  checkpoint
        check_out.append(check_new)

    # return
    return check_out


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


def check_equivalence(check1, check2):
    """
    returns True if check1 and check2 are equivalent
    if one layer is not equivalent, returns False
    """
    equive = True
    for key in check1.keys():
        if not torch.allclose(check1[key], check2[key]):
            equive = False
            break
    return equive

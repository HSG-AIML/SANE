import torch
import copy
import collections

import ray
from .progress_bar import ProgressBar
import json

import logging


### test_checkpoint_for_nan ##################################################################################################
def test_checkpoint_for_nan(checkpoint: collections.OrderedDict):
    """
    investigates checkpoint for NaN values.
    Returns True if NaN is found, False otherwise.
    iterates over keys in ordered dict and evaluates the tensors.
    """
    # iterate over modules
    for key in checkpoint.keys():
        if torch.isnan(checkpoint[key]).any():
            return True
    return False


def test_checkpoint_with_threshold(checkpoint, threshold):
    """
    tests if absolute scalar values in checkpoint are higher than threshold
    Returns True if at least one absolute value is > threshold, False otherwise
    """
    # if threshold is inf -> no need to test
    if torch.isinf(torch.tensor(threshold)):
        return False
    # iterate over modules
    for key in checkpoint.keys():
        w = checkpoint[key]
        # check if any absolute value is larger than threshold
        if (w.abs() > threshold).any():
            return True
    return False


def get_net_epoch_lst_from_label(labels):
    trainable_id = []
    trainable_hash = []
    epochs = []
    permutations = []
    handle = []
    for lab in labels:
        id, hash, epoch, perm_id, hdx = get_net_epoch_from_label(lab)
        trainable_id.append(id)
        trainable_hash.append(hash)
        epochs.append(epoch)
        permutations.append(perm_id)
        handle.append(hdx)
    return trainable_id, trainable_hash, epochs, permutations, handle


def get_net_epoch_from_label(lab):
    # print(lab)
    # remove front stem
    tmp1 = lab.split("#_#")
    # extract trial / net ID
    tmp2 = tmp1[0].split("_trainable_")
    tmp3 = tmp2[1].split("_")
    id = tmp3[0]
    # hash has the 10 digits before first #_#
    tmp4 = tmp1[0]
    hash = tmp4[-10:]
    # extract epoch
    tmp4 = tmp1[1].split("_")
    epoch = tmp4[1]
    # extract layer_lst
    # tmp5 = tmp1[2].split('_')
    # layer_lst = tmp5[1]
    # extract permutation_id
    try:
        tmp6 = tmp1[3].split("_")
        perm_id = tmp6[1]
    except Exception as e:
        # print(e)
        perm_id = -999
    handle = f"net_{id}-ep_{epoch}-perm_{perm_id}"

    return id, hash, epoch, perm_id, handle


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def tokenize_checkpoint(
    checkpoint, tokensize: int, return_mask: bool = True, ignore_bn=False
):
    """
    transforms a checkpoint into a sequence of tokens, one token per channel / neuron
    Tokensize can be set to 0 to automatically discover the correct size (maximum) size
    if tokensize is smaller than the maximum size, the tokens will be chunked into tokens of size tokensize
    tokens are zero-padded to tokensize
    masks indicate with 1 where the original tokens were, and 0 where the padding is
    Args:
        checkpoint: checkpoint to be vectorized
        tokensize: int output dimension of each token
        return_mask: bool wether to return the mask of nonzero values
    Returns
        tokens: list of tokens or zero padded tensor of tokens
        mask: mask of nonzero values
        pos: tensor with 3d positions for every token in the vectorized model sequence
    """
    # init output
    tokens = []
    pos = []
    masks = []

    #### Discover Tokensize ####################################################
    if tokensize == 0:
        # discover tokensize
        tokensize = 0
        for key in checkpoint.keys():
            # get valid layers
            # check for batchnorm layers
            if "bn" in key or "downsample.1" in key or "batchnorm" in key:
                # ignore all batchnorm layers if ignore_bn is set
                if ignore_bn:
                    continue
                # otherwise check for other keys in all remaining layers
            # get weights of all layers
            if "weight" in key:
                tmp = checkpoint[key].shape
            # get running mean and var of batchnorm layers
            elif "running_mean" in key or "running_var" in key:
                tmp = checkpoint[key].shape
            else:
                continue
            tempsize = torch.prod(torch.tensor(tmp)) / tmp[0]
            # cat biases to channels if they exist in checkpoint
            if key.replace("weight", "bias") in checkpoint:
                tempsize += 1

                if tempsize > tokensize:
                    tokensize = tempsize
        # for key in checkpoint.keys():
        #     if "weight" in key:
        #         # get correct slice of modules out of vec sequence
        #         if ignore_bn and ("bn" in key or "downsample.1" in key):
        #             continue
        #         tmp = checkpoint[key].shape
        #         tempsize = torch.prod(torch.tensor(tmp)) / tmp[0]
        #         # cat biases to channels if they exist in checkpoint
        #         if key.replace("weight", "bias") in checkpoint:
        #             tempsize += 1

        #         if tempsize > tokensize:
        #             tokensize = tempsize

    # get raw tokens and positions
    tokensize = int(tokensize)

    #### Get Tokens ####################################################
    idx = 0
    # use only weights and biases
    for key in checkpoint.keys():
        # if "weight" in key:
        #     #### get weights ####
        #     if ignore_bn and ("bn" in key or "downsample.1" in key):
        #         continue
        # get valid layers
        # check for batchnorm layers
        if "bn" in key or "downsample.1" in key or "batchnorm" in key:
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key or "running_mean" in key or "running_var" in key:
            w = checkpoint[key]
            # flatten to out_channels x n
            w = w.view(w.shape[0], -1)
            # cat biases to channels if they exist in checkpoint
            if "weight" in key:
                if key.replace("weight", "bias") in checkpoint:
                    b = checkpoint[key.replace("weight", "bias")]
                    w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)

            #### infer positions ####
            # infer # of tokens per channel
            a = w.shape[1] // tokensize
            b = w.shape[1] % tokensize
            token_factor = int(a)
            if b > 0:
                token_factor += 1

            # get positions, repeating for parts of the same token (overall position will be different)
            idx_layer = [
                [idx, jdx] for jdx in range(w.shape[0]) for _ in range(token_factor)
            ]
            # increase layer counter
            idx += 1
            # add to overall position
            pos.extend(idx_layer)

            #### tokenize ####
            # if b> 0, weights need to be zero-padded
            if b > 0:
                # start with the mask (1 where there is a weight, 0 for padding)
                mask = torch.zeros(w.shape[0], tokensize * token_factor)
                mask[:, : w.shape[1]] = torch.ones(w.shape)
                # zero pad the end of w in dim=1 so that shape[1] is multiple of tokensize
                w_tmp = torch.zeros(w.shape[0], tokensize * token_factor)
                w_tmp[:, : w.shape[1]] = w
                w = w_tmp
            else:
                mask = torch.ones(w.shape[0], tokensize * token_factor)

            # break along token-dimension
            w = w.view(-1, tokensize)
            mask = mask.view(-1, tokensize).to(torch.bool)

            # extend out with new tokens, zero's (and only entry) is a list
            tokens.append(w)
            masks.append(mask)

    #### postprocessing ####################################################
    # cat tokens / masks
    tokens = torch.cat(tokens, dim=0)
    masks = torch.cat(masks, dim=0)

    # add index tensor over whole sequence
    pos = [(ndx, idx, jdx) for ndx, (idx, jdx) in enumerate(pos)]
    pos = torch.tensor(pos)
    # cast tensor to int16
    if pos.max() > 32767:
        logging.debug(
            f"max position value of {pos.max()} does not fit into torch.int16 range. Change data type"
        )
        pos = pos.to(torch.int)
    else:
        pos = pos.to(torch.int16)

    if return_mask:
        return tokens, masks, pos
    else:
        return tokens, pos


def tokens_to_checkpoint(tokens, pos, reference_checkpoint, ignore_bn=False):
    """
    casts sequence of tokens back to checkpoint
    Args:
        tokens: sequence of tokens
        pos: sequence of positions
        reference_checkpoint: reference checkpoint to be used for shape information
        ignore_bn: bool wether to ignore batchnorm layers
    Returns
        checkpoint: checkpoint with weights and biases
    """
    # make copy to prevent memory management issues
    checkpoint = copy.deepcopy(reference_checkpoint)
    # use only weights and biases
    idx = 0
    for key in checkpoint.keys():
        # if "weight" in key:
        #     # get correct slice of modules out of vec sequence
        #     if ignore_bn and ("bn" in key or "downsample.1" in key):
        #         continue
        if "bn" in key or "downsample.1" in key or "batchnorm" in key:
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key or "running_mean" in key or "running_var" in key:
            # get modules shape
            mod_shape = checkpoint[key].shape

            # get slice for current layer
            idx_channel = torch.where(pos[:, 1] == idx)[0]
            w_t = torch.index_select(input=tokens, index=idx_channel, dim=0)

            # infer length of content
            contentlength = int(torch.prod(torch.tensor(mod_shape)) / mod_shape[0])

            # update weights
            checkpoint[key] = w_t.view(mod_shape[0], -1)[:, :contentlength].view(
                mod_shape
            )

            # check for bias
            if "weight" in key:
                if key.replace("weight", "bias") in checkpoint:
                    checkpoint[key.replace("weight", "bias")] = w_t.view(
                        mod_shape[0], -1
                    )[:, contentlength]

            # if key.replace("weight", "bias") in checkpoint:
            # checkpoint[key.replace("weight", "bias")] = w_t.view(mod_shape[0], -1)[
            #     :, contentlength
            # ]

            # update counter
            idx += 1

    return checkpoint

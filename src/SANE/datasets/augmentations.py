import torch

from torchvision.transforms import RandomErasing

import random

from .dataset_auxiliaries import (
    tokens_to_checkpoint,
    tokenize_checkpoint,
)

from SANE.git_re_basin.git_re_basin import (
    PermutationSpec,
    zoo_cnn_permutation_spec,
    weight_matching,
    apply_permutation,
)

import logging


#############################################################################
class AugmentationPipeline(torch.nn.Module):
    """
    Wrapper around a stack of augmentation modules
    Handles passing data through stack, isolating properties
    """

    def __init__(self, stack, keep_properties: bool = False):
        """
        passes stream of data through stack
        """
        super(AugmentationPipeline, self).__init__()
        self.stack = stack
        if keep_properties:
            self.forward = self._forward_props
        else:
            self.forward = self._forward

    def _forward(self, ddx, mdx, p, props=None):
        # apply stack 1
        out = (ddx, mdx, p)
        for m in self.stack:
            out = m(*out)
        return out

    def _forward_props(self, ddx, mdx, p, props):
        # apply stack 1
        out = (ddx, mdx, p, props)
        for m in self.stack:
            out = m(*out)
        return out


#############################################################################
class TwoViewSplit(torch.nn.Module):
    """ """

    def __init__(
        self,
        stack_1,
        stack_2,
        mode: str = "copy",
        view_1_canon: bool = True,
        view_2_canon: bool = False,
    ):
        """
        splits input stream of ddx, mask, p in two streams
        passes two streams through stack_1, stack_2, respectively
        if mode == "copy", then ddx, mask, p are cloned
        if mode == "permutation", then mask, p are copied, ddx is sliced along first axis to get permuted versions
        """
        super(TwoViewSplit, self).__init__()
        self.stack_1 = stack_1
        self.stack_2 = stack_2

        self.mode = mode
        if self.mode == "copy":
            self.forward = self._forward_copy
        elif self.mode == "permutation":
            self.forward = self._forward_permutation
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

        self.view_1_canon = view_1_canon
        self.view_2_canon = view_2_canon

    def _forward_copy(self, ddx1, mdx1, p1):
        # clone ddx, mdx
        ddx2, mdx2, p2 = (
            ddx1.clone().to(ddx1.device),
            mdx1.clone().to(ddx1.device),
            p1.clone().to(ddx1.device),
        )
        # apply stack 1
        for m in self.stack_1:
            ddx1, mdx1, p1 = m(ddx1, mdx1, p1)
        # apply stack 2
        for m in self.stack_2:
            ddx2, mdx2, p2 = m(ddx2, mdx2, p2)
        return ddx1, mdx1, p1, ddx2, mdx2, p2

    def _forward_permutation(self, ddx1, mdx1, p1):
        # ddx.shape[-3] contains random permutations
        # choose two out of those and slice
        perm_ids = torch.randperm(
            n=ddx1.shape[-3], dtype=torch.int32, device=ddx1.device
        )[:2]
        if self.view_1_canon == True:
            perm_ids[0] = 0
        if self.view_2_canon == True:
            perm_ids[1] = 0
        # slice+clone second sample first
        logging.debug(f"perm_ids: {perm_ids}")
        ddx2 = (
            torch.index_select(ddx1.clone(), -3, perm_ids[1]).squeeze().to(ddx1.device)
        )
        # slice / overwrite first sample
        ddx1 = torch.index_select(ddx1, -3, perm_ids[0]).squeeze().to(ddx1.device)

        # clone mdx, p
        mdx2, p2 = mdx1.clone().to(ddx1.device), p1.clone().to(ddx1.device)

        # apply stack 1
        for m in self.stack_1:
            ddx1, mdx1, p1 = m(ddx1, mdx1, p1)
        # apply stack 2
        for m in self.stack_2:
            ddx2, mdx2, p2 = m(ddx2, mdx2, p2)
        return ddx1, mdx1, p1, ddx2, mdx2, p2


#############################################################################
class PermutationSelector(torch.nn.Module):
    """
    ffcv batches use the first dimension to store random permutations of the same sample
    at inference, these need to be separatered and a single version picked.
    this module does that
    """

    def __init__(
        self,
        mode: str = "identity",
        keep_properties: bool = False,
    ):
        """
        if mode == "random", then random permutation is chosen
        if mode == "canonical", ddx[0] is chosen
        """
        super(PermutationSelector, self).__init__()
        self.keep_properties = keep_properties
        self.mode = mode
        if self.mode == "random":
            self.forward = self._forward_random
        elif self.mode == "canonical":
            self.forward = self._forward_canonical
        elif self.mode == "identity":
            self.forward = self._forward_identity
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

    def _return(self, ddx, mdx, p, props):
        if self.keep_properties:
            return ddx, mdx, p, props
        else:
            return ddx, mdx, p

    def _forward_canonical(self, ddx, mdx, p, props=None):
        # ddx.shape[0] contains random permutations
        # choose first out of those and slice
        # ddx = ddx[0]
        ddx = (
            torch.index_select(ddx, -3, torch.tensor(0).to(torch.int32))
            .squeeze()
            .to(ddx.device)
        )
        return self._return(ddx, mdx, p, props)

    def _forward_random(self, ddx, mdx, p, props=None):
        # ddx.shape[0] contains random permutations
        # choose one out of those and slice
        # -3 is the index of permutations
        perm_ids = torch.randperm(
            n=ddx.shape[-3], dtype=torch.int32, device=ddx.device
        )[:1]
        logging.debug(f"perm_ids: {perm_ids}")
        # ddx = ddx[perm_ids[0]]
        ddx = torch.index_select(ddx, -3, perm_ids[0]).squeeze().to(ddx.device)

        return self._return(ddx, mdx, p, props)

    def _forward_identity(self, ddx, mdx, p, props=None):
        # pass through data without change
        return self._return(ddx, mdx, p, props)


#############################################################################
class WindowCutter(torch.nn.Module):
    """
    cuts random window chunks out of sequence of tokens
    Args:
        windowsize: size of window
    Returns:
        ddx: torch.tensor sequence of weight/channel tokens
        mdx: torch.tensor sequence of mask tokens
        p: torch.tensor sequence of positions
    """

    def __init__(self, windowsize: int = 12, keep_properties: bool = False):
        super(WindowCutter, self).__init__()
        self.windowsize = windowsize
        self.keep_properties = keep_properties

    def forward(self, ddx, mdx, p, props=None):
        """
        #TODO
        """
        # get lenght of token sequence
        max_len = ddx.shape[-2]
        # sample start
        if max_len == self.windowsize:
            idx_start = 0
        else:
            idx_start = random.randint(0, max_len - self.windowsize)
        idx_end = idx_start + self.windowsize

        # get index tensor
        idx = torch.arange(start=idx_start, end=idx_end, device=ddx.device)

        # apply window
        ddx = torch.index_select(ddx, -2, idx)
        mdx = torch.index_select(mdx, -2, idx)
        p = torch.index_select(p, -2, idx)

        if self.keep_properties:
            return ddx, mdx, p, props
        return ddx, mdx, p


#############################################################################
class MultiWindowCutter(torch.nn.Module):
    """
    cuts k random window chunks out of one sample of sequence of tokens
    Args:
        windowsize: size of window
        k: number of windows. k=1 is equivalent to WindowCutter. Rule of thumb can be: lenght of sequence / windowsize to get full coverage of sample
    Returns:
        ddx: torch.tensor sequence of weight/channel tokens
        mdx: torch.tensor sequence of mask tokens
        p: torch.tensor sequence of positions
    """

    def __init__(self, windowsize: int = 12, k: int = 10):
        super(MultiWindowCutter, self).__init__()
        self.windowsize = windowsize
        self.k = k

    def forward(self, ddx, mdx, p):
        # get lenght of token sequence

        # single sample case: match batch dimension
        if len(ddx.shape) == 2:
            ddx = ddx.unsqueeze(dim=0)
            mdx = mdx.unsqueeze(dim=0)
            p = p.unsqueeze(dim=0)

        # get max index
        max_idx = ddx.shape[1] - self.windowsize + 1

        # draw k random start indices
        idx_starts = torch.randint(0, max_idx, (self.k,))

        # apply slicing
        ddx = [
            ddx[:, idx_start : idx_start + self.windowsize, :]
            for idx_start in idx_starts
        ]
        mdx = [
            mdx[:, idx_start : idx_start + self.windowsize, :]
            for idx_start in idx_starts
        ]
        p = [
            p[:, idx_start : idx_start + self.windowsize, :] for idx_start in idx_starts
        ]

        # cat along batch dimension
        ddx = torch.cat(ddx, dim=0)
        mdx = torch.cat(mdx, dim=0)
        p = torch.cat(p, dim=0)

        # return
        return ddx, mdx, p


class StackBatches(torch.nn.Module):
    """
    stack batches from multi-window cutter to regular batches
    """

    def __init__(
        self,
    ):
        super(StackBatches, self).__init__()

    def forward(self, ddx, mdx, p):
        # stack along first two dimensions
        ddx = ddx.view((ddx.shape[0] * ddx.shape[1], ddx.shape[2], ddx.shape[3]))
        mdx = mdx.view((mdx.shape[0] * mdx.shape[1], mdx.shape[2], mdx.shape[3]))
        p = p.view((p.shape[0] * p.shape[1], p.shape[2], p.shape[3]))
        return ddx, mdx, p


#############################################################################
class ErasingAugmentation(torch.nn.Module):
    """
    #TODO
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: tuple = (0.02, 0.33),
        ratio: tuple = (0.3, 3.3),
        value=0,
    ):
        super(ErasingAugmentation, self).__init__()
        self.re = RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=True
        )

    def forward(self, ddx, mdx, p):
        """
        #TODO
        """
        # unsquezee along channel dimension to match torch random erasing logic
        ddx = ddx.unsqueeze(dim=-3)
        # apply inplace erasing
        self.re(ddx)
        # squeeze back again
        ddx = ddx.squeeze()
        return ddx, mdx, p


#############################################################################
class NoiseAugmentation(torch.nn.Module):
    """ """

    def __init__(self, sigma: float = 0.1, multiplicative_noise: bool = True):
        super(NoiseAugmentation, self).__init__()
        self.sigma = sigma
        if multiplicative_noise:
            self.forward = self._forward_multiplicative
        else:
            self.forward = self._forward_additive

    def _forward_multiplicative(self, ddx, mdx, p):
        ddx = ddx * (1.0 + self.sigma * torch.randn(ddx.shape, device=ddx.device))
        return ddx, mdx, p

    def _forward_additive(self, ddx, mdx, p):
        ddx = ddx + self.sigma * torch.randn(ddx.shape, device=ddx.device)
        return ddx, mdx, p


#############################################################################

import copy
import torch
import ray

from SANE.datasets.progress_bar import ProgressBar


class PermutationAugmentation(torch.nn.Module):
    """ """

    def __init__(
        self,
        ref_checkpoint,
        permutation_number,
        perm_spec,
        tokensize,
        ignore_bn,
        windowsize,
        permutations_per_sample: int = 1,
        num_threads=6,
    ):
        super(PermutationAugmentation, self).__init__()
        self.permutation_number = permutation_number
        self.tokensize = tokensize
        self.windowsize = windowsize
        self.perms_per_sample = permutations_per_sample
        # precompute permutations
        perms, _ = precompute_permutations(
            ref_checkpoint,
            permutation_number,
            perm_spec,
            tokensize,
            ignore_bn,
            num_threads=6,
        )
        self.perms = perms

    def forward(self, ddx, mdx, p):
        if len(ddx.shape) != 2:
            raise NotImplementedError(
                "PermutationAugmentation so far only works with single samples"
            )
            # TODO: implement batched version
            # permutation needs to be adapted, (unless it's ok to apply same  perm/window on all samples in batch)
            # window slicing  of mask and pos needs to be adapted

        # fix window
        max_len = ddx.shape[0]
        idx_start = random.randint(0, max_len - self.windowsize)

        # get random permutation
        perm_ids = torch.randperm(
            n=self.permutation_number, dtype=torch.int32, device="cpu"
        )[: self.perms_per_sample]

        # include ddx original in the list
        # add original ddx to list of permutations
        ddxp = [ddx[idx_start : idx_start + self.windowsize, :]]
        for pdx in perm_ids:
            perm = self.perms[pdx]
            wdx = permute_model_vector(
                wdx=ddx, idx_start=idx_start, window=self.windowsize, perm=perm
            )
            ddxp.append(wdx)
        ddx = torch.stack(ddxp, dim=0)

        # remove potential leading dimensions in case of k=1
        ddx = ddx.squeeze()

        # slice mask and pos
        mdx = mdx[idx_start : idx_start + self.windowsize, :]
        p = p[idx_start : idx_start + self.windowsize, :]

        return ddx, mdx, p


def precompute_permutations(
    ref_checkpoint, permutation_number, perm_spec, tokensize, ignore_bn, num_threads=6
):
    logging.info("start precomputing permutations")
    model_curr = ref_checkpoint
    # find permutation of model to itself as reference
    reference_permutation = weight_matching(
        ps=perm_spec, params_a=model_curr, params_b=model_curr
    )

    logging.info("get random permutation dicts")
    # compute random permutations
    permutation_dicts = []
    for ndx in range(permutation_number):
        perm = copy.deepcopy(reference_permutation)
        for key in perm.keys():
            # get permuted indecs for current layer
            perm[key] = torch.randperm(perm[key].shape[0]).float()
        # append to list of permutation dicts
        permutation_dicts.append(perm)

    logging.info("get permutation indices")
    """
        1: create reference tokenized checkpoints with two position indices
        - position of token in the sequence
        - position of values within the token (per token)
        
        2: map those back to checkpoints
        
        3: apply permutations on checkpoints
        
        4: tokenize the permuted checkpoints again
        
        (5: at getitem: apply permutation on tokenized checkpoints)
        
        """
    # 1: get reference checkpoint
    ref_checkpoint = copy.deepcopy(model_curr)
    ## tokenize reference
    ref_tok_global, positions = tokenize_checkpoint(
        checkpoint=ref_checkpoint,
        tokensize=tokensize,
        return_mask=False,
        ignore_bn=ignore_bn,
    )

    seqlen, tokensize = ref_tok_global.shape[0], ref_tok_global.shape[1]

    # global index on flattened vector
    ref_tok_global = torch.arange(seqlen * tokensize)

    # view in original shape
    ref_tok_global = ref_tok_global.view(seqlen, tokensize)
    print(ref_tok_global.shape)

    # 2: map reference positions
    ref_checkpoint_global = tokens_to_checkpoint(
        tokens=ref_tok_global,
        pos=positions,
        reference_checkpoint=ref_checkpoint,
        ignore_bn=ignore_bn,
    )

    # 3: apply permutations on checkpoints
    ray.init(num_cpus=num_threads)
    pb = ProgressBar(total=permutation_number)
    pb_actor = pb.actor
    # get permutations
    permutations_global = []
    for perm_dict in permutation_dicts:
        perm_curr_global = compute_single_perm.remote(
            reference_checkpoint=ref_checkpoint_global,
            permutation_dict=perm_dict,
            perm_spec=perm_spec,
            tokensize=tokensize,
            ignore_bn=ignore_bn,
            pba=pb_actor,
        )

        permutations_global.append(perm_curr_global)

    permutations_global = ray.get(permutations_global)

    ray.shutdown()

    # cast to torch.int
    permutations_global = [perm_g.to(torch.int) for perm_g in permutations_global]

    return permutations_global, permutation_dicts


@ray.remote(num_returns=1)
def compute_single_perm(
    reference_checkpoint, permutation_dict, perm_spec, tokensize, ignore_bn, pba
):
    # copy reference checkpoint
    index_check = copy.deepcopy(reference_checkpoint)
    # apply permutation on checkpoint
    index_check_perm = apply_permutation(
        ps=perm_spec, perm=permutation_dict, params=index_check
    )
    # vectorize
    index_perm, _ = tokenize_checkpoint(
        checkpoint=index_check_perm,
        tokensize=tokensize,
        return_mask=False,
        ignore_bn=ignore_bn,
    )
    # update counter
    pba.update.remote(1)
    # return list
    return index_perm


def permute_model_vector(wdx, idx_start, window, perm):
    # get slice window on token sequence
    idx_end = idx_start + window
    tokensize = perm.shape[1]

    # slice permutation in tokenized shape
    perm = perm[idx_start:idx_end]

    # flatten perms
    perm = perm.view(-1)

    # slice permutation out of flattened weight tokens
    wdx = torch.index_select(input=wdx.view(-1), index=perm, dim=0)

    # reshape weights
    wdx = wdx.view(window, tokensize)

    # return tokens
    return wdx


class CheckpointAugmentationPipeline(torch.nn.Module):
    """
    Takes a checkpoint and a property tensor and returns a tokenized (batch of permuted) samlpes
    Args:
        perm_spec: permutation specification
        tokensize: size of tokens
        ignore_bn: whether to ignore batchnorm layers
        permutation_number: number of permutations to be computed
        windowsize: size of window to be permuted
    Returns:
        ddx: permuted, tokenized, sliced checkpoint
        mdx: corresponding mask
        p: position indices
        props: property tensor
    """

    def __init__(
        self,
        perm_spec: PermutationSpec,
        tokensize: int = 64,
        ignore_bn: bool = False,
        permutation_number: int = 15,
        windowsize: int = 512,
    ):
        super(CheckpointAugmentationPipeline, self).__init__()
        self.perm_spec = perm_spec
        self.tokensize = tokensize
        self.ignore_bn = ignore_bn
        self.permutation_number = permutation_number
        self.windowsize = windowsize

        self.permuter = PermutationCheckpoint(
            permutation_number=permutation_number, perm_spec=perm_spec
        )

        self.tokenizer = TokenizerAugmentation(
            tokensize=self.tokensize, ignore_bn=self.ignore_bn
        )

    def forward(self, checkpoint, props=None):
        """
        Applies n permutations, tokenization and window cutting to a checkpoint
        Args:
            checkpoint: checkpoint to be augmented
            props: property tensor
        Returns:
            ddx: permuted, tokenized, sliced checkpoint
            mdx: corresponding mask
            p: position indices
            props: property tensor
        """
        # get perms
        ddxl = [checkpoint]
        ddxl.extend(self.permuter(checkpoint))

        for idx, ddx in enumerate(ddxl):
            ddx, mdx, p = self.tokenizer(ddx)
            ddxl[idx] = ddx

        # stack ddx
        ddx = torch.stack(ddxl, dim=0)

        # get lenght of token sequence
        max_len = ddx.shape[-2]

        # adjust local windowsize s.t. its not longer than the actual sequence
        windowsize = min(self.windowsize, max_len)
        # sample start
        idx_start = random.randint(0, max_len - windowsize)
        idx_end = idx_start + windowsize

        # get index tensor
        idx = torch.arange(start=idx_start, end=idx_end, device=ddx.device)

        # apply window
        ddx = torch.index_select(ddx, -2, idx)
        mdx = torch.index_select(mdx, -2, idx)
        p = torch.index_select(p, -2, idx)

        return ddx, mdx, p, props


class PermutationCheckpoint(torch.nn.Module):
    """
    Computes #permutation_number of permutations of a checkpoint
    """

    def __init__(
        self,
        permutation_number,
        perm_spec,
    ):
        super(PermutationCheckpoint, self).__init__()
        self.permutation_number = permutation_number
        self.perm_spec = perm_spec

    def forward(self, checkpoint):
        """
        :param checkpoint: checkpoint to be permuted
        :return: list of permuted checkpoints
        """
        # get reference checkpoint
        # find permutation of model to itself as reference
        reference_permutation = weight_matching(
            ps=self.perm_spec, params_a=checkpoint, params_b=checkpoint
        )

        # compute random permutations
        permutation_dicts = []
        for ndx in range(self.permutation_number):
            perm = copy.deepcopy(reference_permutation)
            for key in perm.keys():
                # get permuted indecs for current layer
                perm[key] = torch.randperm(perm[key].shape[0]).float()
            # append to list of permutation dicts
            permutation_dicts.append(perm)

        # apply permutations on checkpoints
        checkpoints = []
        for perm_dict in permutation_dicts:
            # copy reference checkpoint
            index_check = copy.deepcopy(checkpoint)
            # apply permutation on checkpoint
            index_check_perm = apply_permutation(
                ps=self.perm_spec, perm=perm_dict, params=index_check
            )
            checkpoints.append(index_check_perm)

        return checkpoints


class TokenizerAugmentation(torch.nn.Module):
    """
    Transforms a checkpoint into a tokenized checkpoint
    Args:
        tokensize: size of the token
        ignore_bn: whether to ignore batchnorm layers
    Returns:
        ddx: tokenized weights
        mdx: tokenized masks
        pos: position indices
    """

    def __init__(self, tokensize, ignore_bn=False):
        super(TokenizerAugmentation, self).__init__()
        self.tokensize = tokensize
        self.ignore_bn = ignore_bn

    def forward(self, checkpoint):
        ddx, mdx, pos = tokenize_checkpoint(
            checkpoint, self.tokensize, return_mask=True, ignore_bn=self.ignore_bn
        )
        return ddx, mdx, pos


#############################################################################
class DDPMSelector(torch.nn.Module):
    """
    Keeps tokens, throws away masks for DDPM model
    """

    def __init__(self, keep_properties: bool = False):
        super(DDPMSelector, self).__init__()
        if keep_properties:
            self.forward = self._forward_props
        else:
            self.forward = self._forward

    def _forward(self, ddx, mdx, p, props=None):
        return ddx, p

    def _forward_props(self, ddx, mdx, p, props=None):
        # apply stack 1
        return ddx, p, props


#############################################################################
class NumpyTransformation(torch.nn.Module):
    """
    Wrapper around a stack of augmentation modules
    Handles passing data through stack, isolating properties
    """

    def __init__(
        self,
    ):
        """
        passes stream of data through stack
        """
        super(NumpyTransformation, self).__init__()

    def forward(self, ddx, mdx, p, props=None):
        # apply stack 1
        ddx = ddx.numpy()
        mdx = mdx.numpy()
        p = p.numpy()
        props = props.numpy()
        return (ddx, mdx, p, props)

import torch


def haloify(w_in, pos, windowsize, halosize):
    """
    slices full sequences w and pos into snipets of content winowsize with context 'halo' of halosize
    returns batch of snippets
    """
    assert (
        halosize < windowsize
    ), f"halosize {halosize} should be smaller than windowsize {windowsize}"
    # init output
    w_out = []
    pos_out = []
    # get number of windows
    idx_max = w_in.shape[-2]
    n_windows, res = divmod(idx_max, windowsize)
    if res != 0:
        n_windows += 1
    # print(f'sequencelength: {idx_max} windowsize:{windowsize} n_windows: {n_windows}')

    # print(f'w: {w.shape} sequencelength: {idx_max} windowsize:{windowsize} n_windows: {n_windows}')
    # iterate over windows
    for idx in range(n_windows):
        # case 1: first window: double context (or whatever fits) at the end
        if idx == 0:
            idx_start = 0
            idx_end = min(idx_start + windowsize + 2 * halosize, idx_max)
        # case 2: last window: as much content as fits, fill up with context so that overall length = 2xhalo+windowsize
        elif idx == n_windows - 1:
            idx_start = max(0, idx_max - windowsize - 2 * halosize)
            idx_end = idx_max
        # case 3: any other window: halo + window + halo
        else:
            # start is idx*windowsize (those are not overlapping) - halosize
            idx_start = idx * (windowsize) - halosize
            idx_end = (idx + 1) * (windowsize) + halosize
        # create slice
        index_slice = torch.arange(idx_start, idx_end)
        # if idx == 0 or idx == 1 or (idx == n_windows-2) or (idx==n_windows-1):
        #     print(f"{idx} start:{index_slice[0]} end:{index_slice[-1]}")
        # check conditions
        assert (
            index_slice.shape[0] == windowsize + 2 * halosize
        ), f"index_slice {index_slice.shape} should have shape windowsize+2*halosize {windowsize+2*halosize}"
        assert (
            index_slice[-1] < idx_max
        ), f"window {idx+1}/{n_windows} index_slice {index_slice[-1]} should be smaller than idx_max {idx_max} with windowsize {windowsize} and halosize {halosize}"
        # slice inputs
        w_tmp = torch.index_select(input=w_in, dim=-2, index=index_slice)
        pos_tmp = torch.index_select(input=pos, dim=-2, index=index_slice)

        w_out.append(w_tmp)
        pos_out.append(pos_tmp)

    # stack
    w_out = torch.stack(w_out, dim=-3)
    p_out = torch.stack(pos_out, dim=-3)
    return w_out, p_out


def dehaloify(toks, poss, windowsize, halosize, orig_seqlen):
    """
    maps sequences of snippets with halo back to full sequences
    """
    assert (
        halosize < windowsize
    ), f"halosize {halosize} should be smaller than windowsize {windowsize}"
    # init output
    w_out = []
    pos_out = []
    # get lenght of snippet sequences
    idx_max = toks.shape[-2]
    # get number of windows
    n_windows, res = divmod(orig_seqlen, windowsize)

    if res != 0:
        n_windows += 1
    # print(f'sequencelength: {orig_seqlen} windowsize:{windowsize} n_windows: {n_windows}')

    # iterate over windows
    for idx in range(n_windows):
        # identify slices of content, ignore context (inverse of above)
        # case 1: first window: double context (or whatever fits) at the end
        if idx == 0:
            # first slice: content is exactly the window
            idx_start = 0
            idx_end = windowsize
        # case 2: last window: as much content as fits, fill up with context so that overall length = 2xhalo+windowsize
        elif idx == n_windows - 1:
            # infer lenght of last window
            length = windowsize
            if res != 0:
                length = res
            # get start and end of content from lenght
            idx_start = idx_max - length
            idx_end = idx_max
        # case 3: any other window: halo + window + halo
        else:
            # in the middle, snippets are padded around the content
            idx_start = halosize
            idx_end = halosize + windowsize
        # create slice
        index_slice = torch.arange(idx_start, idx_end)
        # if idx == 0 or idx == 1 or (idx == n_windows-2) or (idx==n_windows-1):
        # print(f"{idx} start:{index_slice[0]} end:{index_slice[-1]}")
        # check conditions
        if not idx == n_windows - 1:
            assert (
                index_slice.shape[0] == windowsize
            ), f"index_slice {index_slice.shape} should have shape windowsize {windowsize}"
        # slice inputs
        w_tmp = torch.index_select(input=toks[:, idx], dim=-2, index=index_slice)
        pos_tmp = torch.index_select(input=poss[:, idx], dim=-2, index=index_slice)

        w_out.append(w_tmp)
        pos_out.append(pos_tmp)

    # stack
    w_out = torch.cat(w_out, dim=-2)
    p_out = torch.cat(pos_out, dim=-2)

    return w_out, p_out

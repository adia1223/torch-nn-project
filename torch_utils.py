import torch


def combine_dimensions(x: torch.Tensor, dim_begin: int, dim_end: int):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


def flip_dimension(x: torch.Tensor, dim: int):
    idx = [i for i in range(x.size(dim) - 1, -1, -1)]
    idx = torch.tensor(idx).to(x.device)
    return x.index_select(dim=dim, index=idx).to(x.device)

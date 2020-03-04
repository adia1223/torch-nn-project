import torch


def combine_dimensions(x: torch.Tensor, dim_begin: int, dim_end: int):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)

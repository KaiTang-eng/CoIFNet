import torch


def generate_mask(data: torch.Tensor, miss_rate: float) -> torch.Tensor:
    if miss_rate == 0:
        return torch.ones_like(data)
    else:
        return torch.round(torch.rand_like(data) + 0.5 - miss_rate).abs()

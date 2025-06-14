import torch

from .block_missing import (
    generate_block_missing_mask,
    generate_block_missing_mask_at_random,
)
from .mar import generate_mask as gen_mar


def generate_mask(data: torch.Tensor, mask_cfg) -> torch.Tensor:
    miss_rate = mask_cfg.rate
    miss_type = mask_cfg.type

    if miss_rate == 0:
        return torch.ones_like(data)

    if miss_type == "MAR":
        return gen_mar(data, miss_rate)
    elif miss_type == "BlockF":
        mask = generate_block_missing_mask(
            data,
            miss_rate=miss_rate,
            block_len=mask_cfg.block_len,
            block_width=mask_cfg.block_width,
        )
        return mask
    elif miss_type == "BlockR":
        return generate_block_missing_mask_at_random(
            data,
            miss_rate=miss_rate,
            max_block_len=mask_cfg.block_len,
            max_block_width=mask_cfg.block_width,
        )
    else:
        raise ValueError(f"Invalid missing type: {miss_type}")

import os
import pickle as pkl
import random
from typing import Optional

import numpy as np
import torch


def update_path(cfg):
    data_path = os.environ.get("DATA_PATH")
    mlflow_tag = os.environ.get("MLFLOW_TAG")
    if data_path is not None:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.data_root = data_path
            cfg.mlflow.tags.label = mlflow_tag
    return cfg


def seed_everything(seed: int) -> None:
    """
    Seed everything
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def generate_seed(n_seeds: int):
    return list(range(n_seeds))


def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pkl(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)


class ResultSaver:
    def __init__(self, path, save=True, logger=None):
        self.path = path
        self.real = []
        self.pred = []
        self.x = []
        self.x_m = []
        self.save = save
        self.log = logger

    def update(self, true, pred, x, x_m):
        if self.save:
            self.real.append(true.detach().cpu().numpy())
            self.pred.append(pred.detach().cpu().numpy())
            self.x.append(x.detach().cpu().numpy())
            self.x_m.append(x_m.detach().cpu().numpy())

    def save_results(self):
        if self.save:
            self.real = np.concatenate(self.real, axis=0)
            self.pred = np.concatenate(self.pred, axis=0)
            self.x = np.concatenate(self.x, axis=0)
            self.x_m = np.concatenate(self.x_m, axis=0)
            np.savez_compressed(
                self.path,
                real=self.real,
                pred=self.pred,
            )


def cal_num_parmas(model):
    return sum(p.numel() for p in model.parameters())


def cal_num_trainable_parmas(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_FLOPs(flops: float, unit: Optional[str] = None):
    if unit is None:
        if flops < 1e3:
            unit = ""
        elif flops < 1e6:
            unit = "K"
        elif flops < 1e9:
            unit = "M"
        elif flops < 1e12:
            unit = "G"
        else:
            unit = "T"
    if unit == "":
        return f"{flops:.2f}"
    else:
        return f"{flops / 10**(3 * ('KMGT'.index(unit) + 1)):.2f}{unit}"


def computational_analytics(model, inputs):
    if type(inputs) is dict:
        new = {}
        for k, v in inputs.items():
            new[k] = v[0:1]
        inputs = new
    else:
        raise TypeError("inputs must be dict")
    from thop import profile

    macs, params = profile(model, inputs=(inputs,))
    return {
        "macs": f"{macs:.0f}",
        "params": f"{params:.0f}",
        "flops": f"{2 * macs:.0f}",
        "F-MACs": format_FLOPs(macs),
        "F-params": format_FLOPs(params),
        "F-flops": format_FLOPs(2 * macs),
    }

# from dataclasses import dataclass
# from pathlib import Path

import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.tools import load_pkl, save_pkl


class DataWapper(Dataset):

    def __init__(
        self, data_source, seq_len: int, pred_len: int, label_len: int = 0
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.is_slice = False
        self.input_x = None
        self._unpack(data_source)

    def _unpack(self, data_source):
        self.data = data_source["data"]
        self.time_feat = data_source["time_feat"]
        self.mask = data_source["mask"]
        self.input = data_source["input"]

    def _set_slice(self, inputs):
        pass

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        if self.is_slice:
            pass
        else:
            return self._idx_get(index)

    def update_input(self, x, y):
        self.input_x = x
        self.input_y = y

    def _idx_get(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        complete_x = self.data[s_begin:s_end]
        complete_y = self.data[r_begin:r_end]
        time_feat_x = self.time_feat[s_begin:s_end]
        time_feat_y = self.time_feat[r_begin:r_end]
        mask_x = self.mask[s_begin:s_end]
        mask_y = self.mask[r_begin:r_end]
        if self.input_x is not None:
            input_x = self.input_x[index]
            input_y = self.input_y[index]
        else:
            input_x = self.input[s_begin:s_end]
            input_y = self.input[r_begin:r_end]

        one = {
            "input_x": input_x,
            "input_y": input_y,
            "complete_x": complete_x,
            "complete_y": complete_y,
            "time_feat_x": time_feat_x,
            "time_feat_y": time_feat_y,
            "mask_x": mask_x,
            "mask_y": mask_y,
        }

        return one


class BaseDataset:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self._set_dataset_attribute(cfg.dataset)

        data_sources = self._get_complete_data()
        data_sources = self._get_miss_data(data_sources)

        self.data_sources = data_sources

    def _generate_data(self):

        cfg = self.cfg

        seq_len = cfg.model.seq_len
        scaler = cfg.scaler
        data, time_feats = self._parse(
            cfg.data_root,
            seq_len,
            scaler_type=scaler,
        )

        data_sources = {}
        for flag in ["train", "valid", "test"]:
            b_l, b_r = self.borders[flag]

            data_sources[flag] = {
                "data": torch.from_numpy(data[b_l:b_r]).float(),
                "time_feat": torch.from_numpy(time_feats[b_l:b_r]).float(),
            }
        data_sources["scaler"] = self.scaler
        return data_sources

    def _set_dataset_attribute(self, C):
        for k, v in C.items():
            setattr(self, k, v)

    def _get_complete_data(self):
        self.file_name = f"{self.name}_{self.features}_{self.time_enc}.pkl"
        data_path = osp.join(self.cfg.data_cache, self.file_name)
        if not osp.exists(data_path):
            data_sources = self._generate_data()
            save_pkl(data_sources, data_path)
        else:
            data_sources = load_pkl(data_path)
        return data_sources

    def _get_miss_data(self, data_sources):
        seed = self.cfg.seed
        miss_rate = self.cfg.miss.rate
        miss_type = self.cfg.miss.type
        cache_fn = f"{self.file_name}_{miss_type}_{miss_rate}_{seed}.pkl"
        mask_path = osp.join(self.cfg.data_cache, cache_fn)

        if not osp.exists(mask_path):
            masks = {}
            for k in data_sources:
                if k != "scaler":
                    from .mask import generate_mask

                    masks[k] = generate_mask(
                        data_sources[k]["data"], self.cfg.miss
                    )
            save_pkl(masks, mask_path)
        else:
            masks = load_pkl(mask_path)

        #
        for _, m in masks.items():
            rate = 1 - m.sum() / m.numel()
            assert miss_rate == round(rate.item(), 1)
        #

        for k in data_sources:
            if k != "scaler":
                data_sources[k]["mask"] = masks[k]
                data_sources[k]["input"] = data_sources[k]["data"] * masks[k]

        return data_sources

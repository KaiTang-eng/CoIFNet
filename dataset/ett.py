import os.path as osp
from typing import Optional

import pandas as pd

from utils.scaler import init_scaler

from .base import BaseDataset
from .timefeatures import time_features


class EttHour(BaseDataset):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _parse(
        self,
        data_root: str,
        seq_len: int,
        scaler_type: Optional[str] = "standard",
    ):
        df_raw = pd.read_csv(
            osp.join(data_root, "ETT-small", f"{self.name}.csv")
        )

        TRAIN_IDX = 12 * 30 * 24
        VALID_IDX = TRAIN_IDX + 4 * 30 * 24
        TEST_IDX = VALID_IDX + 8 * 30 * 24
        self.borders = {
            "train": [0, TRAIN_IDX],
            "valid": [TRAIN_IDX - seq_len, VALID_IDX],
            "test": [VALID_IDX - seq_len, TEST_IDX],
        }

        if self.features == "M" or self.features == "MS":
            cols = df_raw.columns[1:]
            df_data = df_raw[cols]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise NotImplementedError

        if scaler_type:
            self.scaler = init_scaler(scaler_type)
            t_l, t_r = self.borders["train"]
            self.scaler.fit(df_data.values[t_l:t_r])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        df_date = df_stamp["date"]

        if self.time_enc == "simple":
            df_stamp.loc[:, "month"] = df_date.dt.month
            df_stamp.loc[:, "day"] = df_date.dt.day
            df_stamp.loc[:, "hour"] = df_date.dt.hour
            df_stamp.loc[:, "minute"] = df_date.dt.minute.map(
                lambda x: x // 15
            )
            time_feat = df_stamp.drop(columns=["date"]).values
        elif self.time_enc == "complex":
            time_feat = time_features(
                pd.to_datetime(df_date.values), freq=self.freq
            )
        else:
            raise NotImplementedError

        return data, time_feat


class EttMinute(BaseDataset):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _parse(
        self,
        data_root: str,
        seq_len: int,
        scaler_type: Optional[str] = "standard",
    ):
        df_raw = pd.read_csv(
            osp.join(data_root, "ETT-small", f"{self.name}.csv")
        )

        TRAIN_IDX = 12 * 30 * 24 * 4
        VALID_IDX = TRAIN_IDX + 4 * 30 * 24 * 4
        TEST_IDX = VALID_IDX + 4 * 30 * 24 * 4
        self.borders = {
            "train": [0, TRAIN_IDX],
            "valid": [TRAIN_IDX - seq_len, VALID_IDX],
            "test": [VALID_IDX - seq_len, TEST_IDX],
        }

        if self.features == "M" or self.features == "MS":
            cols = df_raw.columns[1:]
            df_data = df_raw[cols]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise NotImplementedError

        if scaler_type:
            self.scaler = init_scaler(scaler_type)
            t_l, t_r = self.borders["train"]
            self.scaler.fit(df_data.values[t_l:t_r])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        df_date = df_stamp["date"]

        if self.time_enc == "simple":
            df_stamp.loc[:, "month"] = df_date.dt.month
            df_stamp.loc[:, "day"] = df_date.dt.day
            df_stamp.loc[:, "hour"] = df_date.dt.hour
            df_stamp.loc[:, "minute"] = df_date.dt.minute.map(
                lambda x: x // 15
            )
            time_feat = df_stamp.drop(columns=["date"]).values
        elif self.time_enc == "complex":
            time_feat = time_features(
                pd.to_datetime(df_date.values), freq=self.freq
            )
        else:
            raise NotImplementedError

        return data, time_feat


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = {
        "seed": 1,
        "data_root": "/home/vr/hostShare/shared/"
        + "Papers/ForecastingWithMissing/data/processed/all_six_datasets",
        "data_cache": "data/",
        "miss_rate": 0.3,
        "dataset": {
            "name": "ETTh1",
            "data_root": "dataset",
            "features": "M",
            "target": "OT",
            "freq": "h",
            "time_enc": "simple",
            "scaler": "standard",
        },
        "model": {"seq_len": 96},
    }
    cfg = SimpleNamespace(**cfg)
    data = EttHour(cfg)

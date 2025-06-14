from torch.utils.data import DataLoader

from .base import DataWapper
from .custom import CustomDataset
from .ett import EttHour, EttMinute


def init_dataset(cfg):
    DATASETS = {
        "ETTh1": EttHour,
        "ETTh2": EttHour,
        "ETTm1": EttMinute,
        "ETTm2": EttMinute,
        "weather": CustomDataset,
        "exchange_rate": CustomDataset,
    }
    dataset = DATASETS[cfg.dataset.name](cfg)
    return dataset


def get_dataloader(cfg) -> tuple:
    dataset = init_dataset(cfg)
    data_sources = dataset.data_sources
    datas = []
    for k in ["train", "valid", "test"]:
        data = DataWapper(
            data_sources[k], cfg.model.seq_len, cfg.model.pred_len
        )

        data = DataLoader(
            data,
            batch_size=cfg.batch_size,
            shuffle=(k == "train"),
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        datas.append(data)
    train, valid, test = datas
    return train, valid, test, data_sources["scaler"]

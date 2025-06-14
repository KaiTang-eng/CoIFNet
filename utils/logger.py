import logging
import os
import threading
from typing import Any, Dict

import hydra
import mlflow
from omegaconf import DictConfig, open_dict


def singleton(cls):
    _instance = {}
    _lock = threading.Lock()

    def _singleton(*args, **kwargs):
        with _lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def flatten_omegaconf(cfg: DictConfig, sep="_"):
    """
    Flatten an OmegaConf config object into
        a dictionary with a single level of depth.
    Args:
        cfg (DictConfig): the OmegaConf config object to flatten.
        sep (str): the separator to use between the keys.
    Returns:
        Dict: a dictionary with a single level of depth.
    """

    def _flatten(cfg, sep, parent_key=""):
        items = {}
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, DictConfig):
                items.update(_flatten(v, sep, new_key))
            else:
                items[new_key] = v
        return items

    return _flatten(cfg, sep)


def round_result(result: dict[str, float], precision: int = 3):
    for key, value in result.items():
        if isinstance(value, float):
            result[key] = round(value, precision)
    return result


@singleton
class Logger:
    def __init__(self, cfg: DictConfig, log_path=None) -> None:
        if log_path is None:
            with open_dict(cfg):
                cfg.log_path = self._get_hydra_log_path()
        else:
            with open_dict(cfg):
                cfg.log_path = log_path
        self.cfg = cfg
        self._init_logger()
        self.mlflow_run = None
        self.flag_cache = {}
        self.accelerator = None

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def _init_logger(self):
        self.logger = logging.getLogger("main")
        self.fp = os.path.join(self.cfg.log_path, "main.log")
        config = {
            "level": logging.INFO,
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "filename": self.fp,
        }
        self.logger.setLevel(config["level"])
        file_handler = logging.FileHandler(config["filename"])
        file_handler.setLevel(config["level"])
        file_handler.setFormatter(logging.Formatter(config["format"]))
        self.logger.addHandler(file_handler)

    def init_mlflow(self):

        cfg = self.cfg.mlflow
        mlflow.set_experiment(cfg.exp_name)
        self.mlflow_run = mlflow.start_run(
            tags=dict(cfg.tags),
            description=cfg.description,
            log_system_metrics=True,
        )
        self.log_artifact(self.fp)
        return self.mlflow_run

    def get_run(self):
        if self.mlflow_run is None:
            self.init_mlflow()
        return self.mlflow_run

    @staticmethod
    def _get_hydra_log_path():
        return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    def info(self, msg: str):
        self.logger.info(msg)

    def info_with_flag(self, flag: str, msg: str):
        if flag not in self.flag_cache:
            self.flag_cache[flag] = 0
            self.logger.info(f"{flag}: {msg}")
        elif self.flag_cache[flag] < 1:
            self.flag_cache[flag] += 1
            self.logger.info(f"{flag}: {msg}")
        else:
            pass

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ):

        for key, value in metrics.items():
            key = f"{prefix}-{key}"
            mlflow.log_metric(key, value, step=step)

    def log_hyperparams(self, params: DictConfig):
        self.info(str(params))
        mlflow.log_params(flatten_omegaconf(params))

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_model(self, model):
        self.info(str(model))
        # mlflow.pytorch.log_model(model, "models")

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

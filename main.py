import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, open_dict

from dataset import get_dataloader
from model import init_model
from trainer.oneStage import ExpOnestage
from utils.logger import Logger
from utils.tools import generate_seed, seed_everything, update_path


def run_experiment(cfg, logger, seed=None):
    if seed is None:
        seed_everything(cfg.seed)
    else:
        seed_everything(seed)
    train, valid, test, scaler = get_dataloader(cfg)

    if scaler is not None:
        scaler.to_cuda()

    model = init_model(cfg, logger)

    trainer = ExpOnestage(model, cfg=cfg, logger=logger)
    trainer.train(train, valid, scaler)
    return trainer.test(test, scaler)


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig):
    cfg = update_path(cfg)
    seed_everything(cfg.seed)
    logger = Logger(cfg)

    with logger.init_mlflow():
        logger.log_hyperparams(cfg)
        seeds = generate_seed(cfg.n_run)
        results = []
        logger.info(seeds)
        for seed in seeds:
            with open_dict(cfg):
                cfg.seed = seed
            with mlflow.start_run(nested=True):
                res = run_experiment(cfg, logger, seed)
            results.append(res)

        logger.info("===== Final Results =====")
        logger.log_metrics(
            {
                "rmse": np.mean([res["rmse"] for res in results]),
                "mae": np.mean([res["mae"] for res in results]),
                "mape": np.mean([res["mape"] for res in results]),
                "mse": np.mean([res["mse"] for res in results]),
            },
            0,
            "test",
        )
        logger.log_metrics(
            {
                "rmse": np.std([res["rmse"] for res in results]),
                "mae": np.std([res["mae"] for res in results]),
                "mape": np.std([res["mape"] for res in results]),
                "mse": np.std([res["mse"] for res in results]),
            },
            0,
            "test_std",
        )


if __name__ == "__main__":
    main()

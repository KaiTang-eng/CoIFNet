import copy
import datetime
import os
import os.path as osp
import time

import mlflow
import numpy as np
import torch
import torchmetrics as tm
from tqdm import tqdm

from utils.logger import Logger, round_result
from utils.tools import ResultSaver, computational_analytics


class Trainer:
    def __init__(self, model, cfg, logger: Logger) -> None:
        super().__init__()
        self.cfg = cfg
        self.log: Logger = logger
        self.model = model
        self.max_epochs = cfg.max_epochs
        self.device = cfg.device
        self.early_stopping = cfg.early_stopping
        self.best_valid_loss = float("inf")
        self.patience = cfg.patience

        self.es_m = cfg.early_stop_metric

    @staticmethod
    def update_progress(pbar, loss):
        pbar.set_description(f"Loss: {loss:.4f}")

    def train(self, train_dataset, valid_dataset=None, scaler=None):
        self.train_loader = train_dataset
        self.valid_loader = valid_dataset
        self.scaler = scaler
        self.break_epoch = False

        self.before_train()
        for self.epoch in range(self.max_epochs):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.break_epoch:
                break
        self.after_train()

    def before_train(self):
        self.log.log_hyperparams(self.cfg)
        self.log.log_model(self.model)
        self.log.log_hyperparams(
            computational_analytics(
                self.model, self._to_device(next(iter(self.train_loader)))
            )
        )

        self.results = {name: 0 for name in ["rmse", "mae", "mape"]}

        self.start_time = time.time()
        self.best_valid_loss = float("inf")
        self.no_improvement = 0
        self.best_model = None
        self.model.train()
        self.set_optimizers()

    def after_train(self):
        elapsed = round(time.time() - self.start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        self.log.info(f"Training time: {elapsed}")
        if self.cfg.is_save_model:
            self.save_model()

    def save_model(self):
        path = osp.join(self.cfg.log_path, f"models-{self.cfg.seed}")
        if osp.exists(path):
            # move to backup
            os.rename(path, path + "_bak")

        name = self.cfg.model.name.lower()
        if "llm" in name or "gpt" in name:
            model: torch.nn.Module = self.model
            # Only save trainable parameters
            state_dict = {
                k: v for k, v in model.state_dict().items() if v.requires_grad
            }
            torch.save(state_dict, path)
        else:
            mlflow.pytorch.save_model(self.model, path)

    def get_loss(self, batch):
        raise NotImplementedError

    def before_epoch(self):
        self.model.train()
        self.epoch_time = time.time()

    def after_batch(self, batch):
        pass

    def run_epoch(self):
        train_loss = []
        with tqdm(
            self.train_loader, total=len(self.train_loader), leave=False
        ) as pbar:
            for i, batch in enumerate(pbar):
                batch = self._to_device(batch)

                losses, _ = self.get_loss(batch)

                if type(losses) is dict:
                    loss = losses["loss"]
                else:
                    loss = losses

                train_loss.append(loss.detach().item())
                self.model_backward(loss)
                self.after_batch(batch)

                _loss = {"loss": np.mean(train_loss)}
                _log = {**_loss, **self.results}

                self.log.log_metrics(_loss, self.epoch * i)

                extra = {}
                for key, value in losses["extra"].items():
                    extra[key] = value.cpu().detach().item()
                self.log.log_metrics(extra, self.epoch * (i + 1))

                pbar.set_postfix(_log)

    def after_epoch(self):
        self.epoch_time = time.time() - self.epoch_time
        self.log.log_metrics({"Epoch time": self.epoch_time}, self.epoch)

        if self.valid_loader is not None:
            self.results = self.evaluate(self.valid_loader, valid=True)
            self.log.log_metrics(self.results, self.epoch, "valid")
            self.log.info(
                f"Epoch: {self.epoch}, Valid: {round_result(self.results)}"
            )

        if self.early_stopping:
            if self.results[self.es_m] < self.best_valid_loss:
                self.best_valid_loss = self.results[self.es_m]
                self.no_improvement = 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            else:
                self.no_improvement += 1
                if self.no_improvement >= self.patience:
                    self.log.info("Early stopping")
                    self.log.log_params({"Early epoch": self.epoch})
                    self.model.load_state_dict(self.best_model)
                    self.break_epoch = True

    def test(self, test_dataset, scaler=None):
        self.model.eval()
        s_time = time.time()
        self.results = self.evaluate(test_dataset, self.cfg.is_save_results)
        e_time = time.time() - s_time
        self.log.log_metrics({"Test time": e_time}, 0)
        self.log.log_metrics(self.results, self.epoch, "test")
        self.log.info(print_test_result(self.results))
        self.log.info(f"Test: {self.results}")
        return copy.deepcopy(self.results)

    def _get_supervised_data(self, batch):
        y = batch["input_y"]
        y_m = batch["mask_y"]
        c_y = batch["complete_y"]
        x = batch["input_x"]
        x_m = batch["mask_x"]
        return y, y_m, c_y, x, x_m

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(self, dataloader=None, is_save=False, valid=False):
        self.model.eval()
        metric = {
            "rmse": tm.MeanSquaredError(squared=False),
            "mse": tm.MeanSquaredError(),
            "mae": tm.MeanAbsoluteError(),
            "mape": tm.MeanAbsolutePercentageError(),
        }
        saver = ResultSaver(
            osp.join(self.cfg.log_path, f"results-{self.cfg.seed}.npz"),
            save=is_save,
            logger=self.log,
        )
        for batch in dataloader:
            batch = self._to_device(batch)

            loss, y_pred = self.get_loss(batch)
            y, y_m, complete_y, x, x_m = self._get_supervised_data(batch)

            y_pred, supervised_y = self.retrieve_target_values(
                valid, y_pred, y, y_m, complete_y
            )

            saver.update(complete_y, y_pred, x, x_m)
            for name, m in metric.items():
                m.update(y_pred.detach().cpu(), supervised_y.detach().cpu())

        saver.save_results()
        for name in metric.keys():
            metric[name] = metric[name].compute().item()
        return metric

    def retrieve_target_values(self, valid, y_pred, y, y_m, complete_y):
        if self.cfg.valid_is_missing:
            if valid:
                supervised_y = y
                y_pred = y_pred * y_m
            else:
                supervised_y = complete_y
        else:
            supervised_y = complete_y
        return y_pred, supervised_y

    def predict(self, dataloader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                _, y_pred = self.get_loss(batch)
                preds.append(y_pred)
        return torch.cat(preds, dim=0).cpu().numpy()

    def set_optimizers(self):
        self.optm, self.scheduler = self.configure_optimizers()

    def configure_optimizers(self):
        raise NotImplementedError

    def model_backward(self, loss):
        self.optm.zero_grad()
        loss.backward()
        self.optm.step()
        # self.scheduler.step() # Uncomment if using a scheduler
        # self.scheduler.step(self.results["mae"])


def print_test_result(result: dict[str, float]) -> str:
    strs = "\n===========Test Score=================\n"
    for key, value in result.items():
        strs += f"\t{key.upper()}\t| {value:.6f}\t\n"
    strs += "======================================"
    return strs

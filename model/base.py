import torch
import torch.nn as nn
import torch.nn.functional as F


class Task(nn.Module):
    def __init__(self, cfg, logger):
        super(Task, self).__init__()
        self.log = logger
        self.cfg = cfg
        self.device = cfg.device

        self.seq_len = cfg.model.seq_len
        self.pred_len = cfg.model.pred_len
        self.loss_type = cfg.model.loss_type

        self.loss_lambda = cfg.model.loss_lambda

        self.log.info(f"Model Name: {cfg.model.name}")

    def get_prediction_loss(self, y_pred, y):
        if self.loss_type == "l1":
            return F.l1_loss(y_pred, y)
        elif self.loss_type == "l2":
            return F.mse_loss(y_pred, y)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(y_pred, y)
        else:
            raise ValueError("loss_type should be l1, l2 or smooth_l1")

    def unpack(self, batch):
        x = batch["input_x"]
        y = batch["input_y"]
        x_m = batch["mask_x"]
        y_m = batch["mask_y"]
        x_t = batch["time_feat_x"]
        y_t = batch["time_feat_y"]

        return x, y, x_m, y_m, x_t, y_t

    @staticmethod
    def get_extra_mask(cond_mask, rate=0.3):
        tot_rate = 1 - cond_mask.sum() / cond_mask.numel()

        if rate > 0:
            tot_rate += rate
        elif rate < 0:
            tot_rate = tot_rate + torch.sigmoid(
                torch.randn(1).to(tot_rate.device)
            ) * (1 - tot_rate)
        elif rate == 0:
            tot_rate = (1 - tot_rate) / 2
        else:
            raise ValueError("rate should be positive or negative")

        mask = (
            cond_mask * (torch.rand_like(cond_mask) + 0.5 - tot_rate).round()
        ).abs()
        return mask

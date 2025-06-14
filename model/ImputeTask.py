import torch
from omegaconf import open_dict

from backbone.imputation import imputation_factory

from .base import Task


class ImputeTask(Task):
    def __init__(self, cfg, log=None):
        super(ImputeTask, self).__init__(cfg)
        self.extra_mask_rate = cfg.model.extra_mask_rate
        self.log = log
        with open_dict(cfg):
            cfg.model.seq_len = cfg.model.seq_len + cfg.model.pred_len
        self.model = imputation_factory(cfg.model.name, cfg)
        self.name = cfg.model.name

        self.to(self.device)

    def forward(self, batch):
        x, y, x_m, y_m, x_t, y_t = self.unpack(batch)

        if not self.model.training:
            super_y = torch.zeros_like(y)
            super_y_m = torch.zeros_like(y_m)
        else:
            super_y = y
            super_y_m = y_m

        super_x = torch.cat([x, super_y], dim=1)
        super_mask = torch.cat([x_m, super_y_m], dim=1)

        if self.name == "BRITS":
            loss_impute, loss_reconstruct, impute = self.model.get_impute_loss(
                super_x,
                super_mask,
                None,
            )
        else:
            loss_impute, loss_reconstruct, _ = self.mit_task(
                super_x, super_mask
            )
            impute = self.impute_task(super_x, super_mask)

        loss = loss_impute + loss_reconstruct
        y_pred = impute[:, x.shape[1] :]

        losses = {
            "loss": loss,
            "extra": {
                "loss_impute": loss_impute,
                "loss_reconstruct": loss_reconstruct,
            },
        }

        return losses, y_pred

    def mit_task(self, x, x_m):
        """
        x: input data(batch_size, seq_len, feature_dim)
        y: target data(batch_size, seq_len, feature_dim)
        """
        extra_mask = self.get_random_mask(x_m, self.cfg.model.extra_mask_rate)
        extra_x = extra_mask * x
        loss_impute, loss_reconstruct, impute_x = self.model.get_impute_loss(
            extra_x, x_m, extra_mask
        )

        return (
            loss_impute,
            loss_reconstruct,
            impute_x,
        )

    def impute_task(self, x, x_m):
        impute_x = self.model(x, x_m)

        return impute_x

import torch

from utils import metric as M

from .base import Task
from .CoIFNet import CoIFNet


class CoIFNetTask(Task):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.use_reconstruct = cfg.model.use_reconstruct
        self.lab = cfg.model.loss_lambda

        self.log = logger
        seq_len, pred_len = cfg.model.seq_len, cfg.model.pred_len

        horizon_len = seq_len + pred_len if self.use_reconstruct else pred_len

        self.model = CoIFNet(cfg, seq_len=seq_len, pred_len=horizon_len)
        self.m_cfg = cfg.model

        self.to(self.device)

    def forward(self, batch, epoch=None):
        self.epoch = epoch

        x, super_y, x_m, super_y_m, x_t, y_t = self._unpack(batch)

        x_f, _ = self._process_timestamps_feat(x_t, y_t)

        in_len = x.size(1)
        impute = self.model.impute(x, x_m, feat=x_f)

        if self.use_reconstruct:
            impute_x, impute_y = (
                impute[:, :in_len],
                impute[:, in_len:],
            )
        else:
            impute_x, impute_y = None, impute

        loss_forecast = M.masked_mae(super_y, impute_y, super_y_m)

        if self.use_reconstruct:
            loss_reconstruct = M.masked_mae(x, impute_x, x_m)
            loss = self.lab * loss_forecast + (1 - self.lab) * loss_reconstruct

            extra = {
                "loss_impute": loss_forecast,
                "loss_reconstruct": loss_reconstruct,
            }
        else:
            loss = loss_forecast
            extra = {"loss_impute": loss_forecast}

        losses = {
            "loss": loss,
            "extra": extra,
        }

        return losses, impute_y

    def get_shared_parameters(self):
        return self.model.get_shared_parameters()

    def _process_timestamps_feat(self, x_feat, y_feat):
        if self.m_cfg.use_feat_emb:
            x_w = (x_feat[:, :, 1:2] % 7).to(int)
            x_h = x_feat[:, :, 2:3].to(int)
            x_f = torch.cat([x_w, x_h], dim=-1)
            y_w = (y_feat[:, :, 1:2] % 7).to(int)
            y_h = y_feat[:, :, 2:3].to(int)
            y_f = torch.cat([y_w, y_h], dim=-1)

            return x_f, y_f
        else:
            return x_feat, y_feat

    def _unpack(self, batch):
        x, y, x_m, y_m, x_t, y_t = super().unpack(batch)
        if not self.model.training:
            super_y = torch.zeros_like(y)
            super_y_m = torch.zeros_like(y_m)
            self.log.info_with_flag("Valid", f"y_m counts: {super_y_m.sum()}")
        else:
            super_y = y
            super_y_m = y_m

        return x, super_y, x_m.abs(), super_y_m.abs(), x_t, y_t

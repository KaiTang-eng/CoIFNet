from backbone.forecast import forecasting_factory

from .base import Task


class MaskTask(Task):

    def __init__(self, cfg, logger=None):
        super(MaskTask, self).__init__(cfg)
        self.logger = logger
        logger.info("Using MaskTask")
        self.extra_mask_rate = cfg.model.extra_mask_rate

        self.model = forecasting_factory(cfg.model.name, cfg)

        self.to(self.device)

    def forward(self, batch):
        (x, y, x_m, y_m, x_t, y_t) = self.unpack(batch)

        y_pred = self.model(x, x_m, x_t)

        loss = self.get_prediction_loss(y_pred * y_m, y)

        losses = {
            "loss": loss,
            "extra": {},
        }

        return losses, y_pred

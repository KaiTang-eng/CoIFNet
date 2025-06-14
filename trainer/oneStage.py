from torch import optim

from .engine import Trainer


class ExpOnestage(Trainer):
    def get_loss(self, batch):
        return self.model(batch, self.epoch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        return optimizer, None
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=20,
        #     T_mult=1,
        #     eta_min=1e-6,
        # )
        # return optimizer, scheduler

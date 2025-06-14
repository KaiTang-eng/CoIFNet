def init_model(cfg, logger):
    if cfg.model.type == "CoIFNet":
        from .CoIFNetTask import CoIFNetTask

        return CoIFNetTask(cfg, logger)

    elif cfg.model.type == "Forecast":
        from .MaskTask import MaskTask

        return MaskTask(cfg, logger)
    elif cfg.model.type == "Imputation":
        from .ImputeTask import ImputeTask

        return ImputeTask(cfg, logger)

    else:
        raise ValueError(f"model type {cfg.model.type} not supported")

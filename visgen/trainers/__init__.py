from .trainer import BaseTrainer  # , BiLevelTrainer, MultiStepTrainer


def get_trainer(cfg):
    trainer_type = cfg.training.get("trainer", "base")
    device = cfg["device"]
    if trainer_type == "base":
        trainer = BaseTrainer(cfg["training"], device=device)
    # elif trainer_type == "multistep":
    #     trainer = MultiStepTrainer(cfg["training"], device=device)
    # elif trainer_type == "bilevel":
    #     trainer = BiLevelTrainer(cfg["training"], device=device)
    else:
        raise ValueError(f"{trainer_type} is not a valid trainer!")
    return trainer

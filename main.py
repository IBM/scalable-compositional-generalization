import argparse
import os

import torch
import wandb
from omegaconf import OmegaConf

from visgen.datasets import get_dataloaders
from visgen.models import get_model
from visgen.trainers import get_trainer
from visgen.utils.general import (compare_cfgs, fix_random,
                                  get_logger, register_resolvers,
                                  save_yaml_safe)
from visgen.utils.general.general import get_lsf_info


def parse_args():
    """Parse CLI arguments.

    Returns:
        (argparse.Namespace, list): returns known and unknown parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cfg", type=str, default="configs/base.yml")
    parser.add_argument("--data-cfg", type=str)
    parser.add_argument("--model-cfg", type=str)
    parser.add_argument("--experiment-cfg", type=str)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--sweep", default=False, action="store_true")
    parser.add_argument("--ed-superposition-init", type=str)
    return parser.parse_known_args()


def custom_cfg_conflict_resolution(cfg, model_cfg, data_cfg, experiment_cfg, cfg_cli):
    # handle batch size merging separately, we need to take the minimum here
    if (
        hasattr(model_cfg, "data")
        and hasattr(model_cfg.data.training, "batch_size")
        and hasattr(data_cfg.data.training, "batch_size")
    ):
        cfg.data.training.batch_size = min(
            data_cfg.data.training.batch_size, model_cfg.data.training.batch_size
        )
        cfg.data.testing.batch_size = min(
            data_cfg.data.testing.batch_size, model_cfg.data.testing.batch_size
        )
    # max epoch setting
    if (
        hasattr(experiment_cfg, "training")
        and hasattr(cfg_data, "training")
        and not (hasattr(cfg_cli, "training") and hasattr(cfg_cli.training, "n_epoch"))
    ):
        # cfg.training.n_epoch = max(data_cfg.training.n_epoch, experiment_cfg.training.n_epoch)
        cfg.training.n_epoch = data_cfg.training.n_epoch
    # preprocessing as part of the model
    if hasattr(model_cfg, "model") and hasattr(model_cfg.model, "preprocessing"):
        cfg.model.preprocessing = model_cfg.model.preprocessing
    return cfg


def log_info(cfg, debug=False):
    # add LSF info, dump them into a different file
    cfg["lsf"] = get_lsf_info()
    save_yaml_safe(os.path.join(cfg.path.full, "lsf.yml"), cfg.lsf)
    # save config file
    cfg_path = os.path.join(cfg.path.full, "cfg.yml")
    if os.path.exists(cfg_path):
        existing_cfg = OmegaConf.load(cfg_path)
        # replace current runid with the existing one, to allow wandb resume
        cfg.logger.run_id = existing_cfg.logger.run_id
        compare_cfgs(cfg, existing_cfg)
        # if not equal and cfg.training.get("if_exists") != "retrain" and not debug:
        #     raise ValueError("Conflicting CFGs!")
    save_yaml_safe(cfg_path, cfg)
    return cfg


if __name__ == "__main__":

    # parse pure CLI args
    args, unknown = parse_args()

    # remove "--" from remaining parameters
    # (mostly to support wandb sweep args)
    unknown = [u.lstrip("--") for u in unknown]

    # parse CLI and yaml Omega args
    cfg_cli = OmegaConf.from_dotlist(unknown)
    cfg_base = OmegaConf.load(args.base_cfg)
    cfg_data = OmegaConf.load(args.data_cfg)
    cfg_model = OmegaConf.load(args.model_cfg)
    cfg_experiment = OmegaConf.load(args.experiment_cfg)

    # merge order is important here, determines the hierarchy
    cfg = OmegaConf.merge(cfg_base, cfg_model, cfg_data, cfg_experiment, cfg_cli)
    # resolve conflicts in configs
    cfg = custom_cfg_conflict_resolution(
        cfg, cfg_model, cfg_data, cfg_experiment, cfg_cli
    )

    # add device cfg
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.logger["sweep"] = args.sweep

    # register custom resolvers for Omega
    register_resolvers()

    # get run name (name of the yaml file)
    if args.debug:
        run_name = "debug"
    else:
        run_name = os.path.basename(args.experiment_cfg).split(".")[0]
    model_name = os.path.basename(args.model_cfg).split(".")[0]

    # if test only, set number of epochs to 0 and turn off wandb
    if args.test:
        cfg.training.n_epoch = 0
        cfg.logger.name = "base"
    run_id = wandb.util.generate_id()
    cfg.logger.run_id = run_id

    # compose the full save path
    model_groupby = cfg.data.training.targets
    cfg.model.path = cfg.path.full = os.path.join(
        cfg.path.base,
        run_name,
        cfg.data.training.dataset,
        # f"{cfg.data.training.split}_{str(cfg.data.training.split_difficulty)}",
        f"{cfg.data.training.split}_{str(cfg.data.training.c)}",
        # cfg.data.training.split,
        model_name,
        model_groupby,
        str(cfg.seed),
    )
    os.makedirs(cfg.path.full, exist_ok=True)

    # log execution information
    cfg = log_info(cfg, args.debug)

    ### FROM HERE, PROPER EXECUTION STARTS ###

    # get logger
    cfg.logger.group = run_name
    writer = get_logger(cfg)

    # run the experiment
    fix_random(cfg.seed)

    # load data
    d_dataloaders = get_dataloaders(cfg.data, writer)

    # build model
    model = get_model(cfg).to(cfg.device)

    # setup trainer
    trainer = get_trainer(cfg)
    model, res = trainer.train(
        model,
        d_dataloaders,
        writer=writer,
        savepath=os.path.join(cfg.path.full, "checkpoints"),
    )

    # final training summary
    # print(beautify_results(res))

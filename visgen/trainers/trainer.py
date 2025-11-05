import os
import time

import numpy as np
import torch
from tqdm import tqdm

from visgen.utils.general import (AverageMeter, load_checkpoint, plot_reconstructed)
from .optimizers import get_optimizer

# brute force switch of image upload
WRITE_IMAGES = True


class BaseTrainer:
    """"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def train(self, model, d_dataloaders, savepath, writer=None, prefix=""):

        # create experiment folder
        os.makedirs(savepath, exist_ok=True)

        # get data loaders
        train_loader, val_loader, test_loader = (
            d_dataloaders["training"],
            d_dataloaders["validation"],
            d_dataloaders["testing"],
        )
        # subtract 3 (train, val, test) to get the number of OOD validation sets
        num_ood_sets = len(d_dataloaders) - 3
        ood_val_loaders = [
            d_dataloaders[f"ood_validation_{i}"] for i in range(num_ood_sets)
        ]
        kwargs = {"dataset_size": len(train_loader.dataset)}
        selection_metric = self.cfg.selection_metric
        best_val_metric = -np.inf if "acc" in selection_metric else np.inf
        if "test_metric" in self.cfg:
            test_metric = self.cfg.test_metric
            best_test_metric = -np.inf if "acc" in test_metric else np.inf
        else:
            test_metric, best_test_metric = None, None
        best_ams = {}

        # init optimizer
        optimizer = get_optimizer(self.cfg["optimizer"], model.parameters())

        # init varia
        metrics = model.get_logged_metrics()
        amp_scaler = (
            torch.amp.GradScaler("cuda") if self.cfg.get("fp-16", False) else None
        )

        # CHECKPOINTING
        # handle training re-start or continuation
        model_best = os.path.join(savepath, "model_best.pth.tar")
        model_last = os.path.join(savepath, "checkpoint.pth.tar")
        os.path.join(savepath, "results.json")
        force_train = self.cfg.get("if_exists", "continue")
        start_epoch = 1

        # Re-train model
        if force_train == "retrain":
            try:
                os.remove(model_last)
                os.remove(model_best)
            except Exception:
                pass

        # Continue training if a model already exists otherwise
        elif os.path.exists(model_last):
            model, best_ams, start_epoch, optimizer = load_checkpoint(
                model_last, model, optimizer, self.device
            )
            best_val_metric = best_ams[prefix + selection_metric]
            print(
                "\n\n",
                f"Found trained model in {model_last},",
                f"from epoch {start_epoch} with best {prefix + selection_metric}: {best_val_metric}.",
                "Loading it and continuing training!",
                "\n\n",
            )

        # plot debug plots before starting training, if the model supports it
        if callable(getattr(model, "plot_debug", None)):
            images, captions = model.plot_debug(next(iter(train_loader))[0], savepath)
            write_images = getattr(writer, "write_images", None)
            if callable(write_images) and WRITE_IMAGES:
                writer.write_images(images, captions)

        # TRAINING LOOP
        for i_epoch in range(start_epoch, self.cfg["n_epoch"] + 1):
            ams = {}
            for split in ["train", "val", "test"] + [
                f"ood_val_{i}" for i in range(num_ood_sets)
            ]:
                ams |= {f"{split}_{m}": AverageMeter(m) for m in metrics}
            start = time.time()

            # training steps
            model.train()
            for x, y in tqdm(train_loader, disable=not self.cfg["verbose"]):
                d_train = model.train_step(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    optimizer=optimizer,
                    amp_scaler=amp_scaler,
                    **kwargs,
                )
                for k, v in d_train.items():
                    ams[f"train_{k}"].update(v)

            # evaluation steps
            model.eval()
            loaders = [
                ("val", val_loader),
                ("test", test_loader),
            ]
            loaders += [
                (f"ood_val_{i}", loader) for i, loader in enumerate(ood_val_loaders)
            ]
            for name, loader in loaders:
                for x, y in tqdm(loader, disable=not self.cfg["verbose"]):
                    d_val = model.validation_step(
                        x=x.to(self.device), y=y.to(self.device)
                    )
                    for k, v in d_val.items():
                        ams[f"{name}_{k}"].update(v)

            # compute OOD val accuracy on all the validation splits
            if selection_metric == "min_ood_val_acc":
                ood_val_accs = [
                    ams[f"ood_val_{i}_acc"].avg for i in range(num_ood_sets)
                ]
                if "min_ood_val_acc" not in ams:
                    ams["min_ood_val_acc"] = AverageMeter("min_ood_val_acc")
                ams["min_ood_val_acc"].update(min(ood_val_accs))
            elif selection_metric == "wio_acc":
                ood_val_accs = [
                    ams[f"ood_val_{i}_acc"].avg for i in range(num_ood_sets)
                ]
                if "wio_acc" not in ams:
                    ams["wio_acc"] = AverageMeter("wio_acc")
                ams["wio_acc"].update(
                    ams["val_acc"].avg + (min(ood_val_accs) - 100) / 10
                )

            if "acc" in selection_metric:
                best_model = ams[selection_metric].avg >= best_val_metric
            else:
                best_model = ams[selection_metric].avg <= best_val_metric

            logams = {prefix + k: v.avg for k, v in ams.items()}
            plot_train = None
            plot_test = None

            if self.cfg["objective"] == "reconstruction":
                # log visualizations
                x_vis_train = next(iter(train_loader))[0].to(self.device)
                x_vis_test = next(iter(test_loader))[0].to(self.device)
                x_vis_train, x_vis_test = model.preprocessing(
                    x_vis_train
                ), model.preprocessing(x_vis_test)
                x_recon_train = model.visualization_step(x_vis_train)["img"]
                x_recon_test = model.visualization_step(x_vis_test)["img"]
                plot_train = plot_reconstructed(x_vis_train, x_recon_train, N=6)
                plot_test = plot_reconstructed(x_vis_test, x_recon_test, N=6)

            writer.write(logams)
            if plot_train is not None and WRITE_IMAGES:
                writer.write_images([plot_train, plot_test], ["train-img", "test-img"])

            if best_model:
                best_val_metric = logams[prefix + selection_metric]
                best_ams = logams
            if test_metric is not None:
                current_metric = logams[prefix + test_metric]
                # record test metric of last epoch
                best_ams[f"{prefix}last_{test_metric}"] = current_metric

                # record best test metric value to compare against value achieved by selected model
                is_accuracy_metric = "acc" in test_metric
                if (is_accuracy_metric and current_metric > best_test_metric) or (
                    not is_accuracy_metric and current_metric < best_test_metric
                ):
                    best_test_metric = current_metric
                best_ams[f"{prefix}best_{test_metric}"] = best_test_metric

            # End of epoch, log results
            print(
                f"Epoch [{i_epoch:d}]\n   "
                + "\n   ".join([f"{k}: {v.avg:.6f}" for k, v in ams.items()])
                + f"\n   elapsed: {(time.time() - start):.1f}"
                + f"\n   best model: {best_model}"
            )
        # END TRAINING LOOP

        # # load best model and save the metrics
        # model, best_ams, _, _ = load_checkpoint(model_best, model, device=self.device)
        # # if checkpointing is not enabled, remove best model checkpoint after loading
        # if not self.cfg["checkpointing"]:
        #     try:
        #         os.remove(model_best)
        #         os.remove(model_last)
        #     except OSError:
        #         pass
        # # log results in separate json file
        # save_json(res_path, best_ams)

        # # post-training logs
        # model.eval()
        # if callable(getattr(model, "confusion_matrix", None)):
        #     images, captions = model.confusion_matrix(test_loader)
        #     if callable(getattr(writer, "write_images", None)) and WRITE_IMAGES:
        #         writer.write_images(images, captions)

        # return model, best_ams
        return (1,1)

class MultiStepTrainer:
    """"""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def train(self, model, d_dataloaders, savepath, writer=None):
        results = {}
        for step in self.cfg.steps:
            name = step["objective"]
            print(f"Step: {name}")
            savepath_step = os.path.join(savepath, name)
            trainer = BaseTrainer(step, self.device)
            model, best_ams = trainer.train(
                model, d_dataloaders, savepath_step, writer, prefix=f"{name}/"
            )
            results = results | best_ams
            model.next_step()
        return model, results


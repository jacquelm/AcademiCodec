from pathlib import Path
import numpy as np
import os
import json
import re
from beartype import beartype

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import get_dataloader, EmbeddingDataset
from .optimzer import get_optimizer
from .loss import get_accuracy, get_loss
from torch.utils import tensorboard
import time
from tqdm import tqdm
from accelerate import (
    Accelerator,
    DistributedType,
    DistributedDataParallelKwargs,
    DataLoaderConfiguration,
)
import torch
from torch import nn

import matplotlib.pyplot as plt


def exists(val):
    return val is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r"\d+", str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


def plot_history(history):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(
        np.arange(0, len(history["loss"]["train"])),
        history["loss"]["train"],
        color="red",
        label="train",
    )
    ax.plot(
        np.arange(0, len(history["loss"]["validation"])),
        history["loss"]["validation"],
        color="blue",
        label="validation",
    )
    ax.set_title("Loss history")
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(
        np.arange(0, len(history["accuracy"]["train"])),
        history["accuracy"]["train"],
        color="red",
        label="train",
    )
    ax.plot(
        np.arange(0, len(history["accuracy"]["validation"])),
        history["accuracy"]["validation"],
        color="blue",
        label="validation",
    )
    ax.set_title("Accuracy history")
    ax.legend()
    fig.tight_layout()
    fig.show()


class EmbeddingTrainer(nn.Module):
    """
    Trainer class for the Embedding model, handling training, validation, saving, and
    loading of the model.
    """

    def __init__(
        self,
        model,
        cfg,
        accelerate_kwargs: dict = dict(),
    ):
        """
        Initialize the EmbeddingTrainer.

        Parameters
        ----------
        model :
            The model model to train.
        cfg : dict
            Configuration dictionary containing hyperparameters and settings.
        accelerate_kwargs : dict, optional
            Additional arguments for the Accelerator (default is an empty dictionary).
        """
        super().__init__()

        # Set up random seed for reproducibility
        torch.manual_seed(cfg.get("seed"))

        # Configuration parameters
        split_batches = cfg.get("split_batches", False)
        self.log_steps = cfg.get("log_steps")
        self.stdout_steps = cfg.get("stdout_steps")
        self.save_model_steps = cfg.get("save_model_steps")
        results_folder = cfg.get("results_folder")
        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = cfg.get("num_ckpt_keep")
        self.epochs = cfg.get("epochs")
        self.num_warmup_steps = cfg.get("num_warmup_steps")
        self.batch_size = cfg.get("batch_size")
        self.sample_rate = cfg.get("sample_rate")

        self.n_mfcc = cfg.get("n_mfcc")
        self.n_mel = cfg.get("n_mel")
        self.n_freq = cfg.get("n_freq")
        self.n_fft = cfg.get("n_fft")
        self.hop_size = cfg.get("hop_size")
        self.win_size = cfg.get("win_size")

        self.db = cfg.get("db")
        self.calib = cfg.get("calib")
        self.volt_max = cfg.get("volt_max")
        self.win = cfg.get("win")
        self.step = cfg.get("step")

        self.audio_type = cfg.get("audio_type")
        self.showpiece_num = cfg.get("showpiece_num", 8)
        self.project = cfg.get("project")
        self.create = cfg.get("create")
        self.feature = cfg.get("feature")
        self.csv_dir = cfg.get("csv_dir", None)
        self.additionnal = cfg.get("additionnal", False)

        # Create results directory if it doesn't exist
        if not self.results_folder.exists():
            self.results_folder.mkdir(parents=True, exist_ok=True)

        # Save configuration to file
        with open(f"{str(self.results_folder)}/config.json", "w+") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        # Setup Accelerator for distributed training
        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[DistributedDataParallelKwargs()],
            **accelerate_kwargs,
        )

        # TensorBoard writer setup (only for the main process)
        if self.is_main:
            self.writer = tensorboard.SummaryWriter(
                os.path.join(results_folder, "logs")
            )

        # Initialize models and other components
        self.model = model
        self.register_buffer("steps", torch.Tensor([0]))

        # Load dataset files
        segment_size = cfg.get("segment_size")
        train_files = cfg.get("train_files")
        batch_size = cfg.get("batch_size")
        self.batch_size = batch_size
        with open(train_files, "r") as f:
            train_file_list = f.readlines()
        valid_files = cfg.get("valid_files")
        with open(valid_files, "r") as f:
            valid_file_list = f.readlines()

        # Create datasets
        self.ds = EmbeddingDataset(
            file_list=train_file_list,
            segment_size=segment_size,
            sample_rate=self.sample_rate,
            audio_type=self.audio_type,
            n_mfcc=self.n_mfcc,
            n_mel=self.n_mel,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            db=self.db,
            calib=self.calib,
            volt_max=self.volt_max,
            win=self.win,
            step=self.step,
            project=self.project,
            create=self.create,
            feature=self.feature,
            csv_dir=self.csv_dir,
            additionnal=self.additionnal,
        )
        self.valid_ds = EmbeddingDataset(
            file_list=valid_file_list,
            segment_size=segment_size,
            sample_rate=self.sample_rate,
            audio_type=self.audio_type,
            n_mfcc=self.n_mfcc,
            n_mel=self.n_mel,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            db=self.db,
            calib=self.calib,
            volt_max=self.volt_max,
            win=self.win,
            step=self.step,
            project=self.project,
            create=self.create,
            feature=self.feature,
            csv_dir=self.csv_dir,
            additionnal=self.additionnal,
            valid=True,
        )
        if self.is_main:
            self.print(
                f"training with dataset of {len(self.ds)} samples and validating with \
                    randomly splitted {len(self.valid_ds)} samples"
            )

        # Ensure datasets are large enough
        assert (
            len(self.ds) >= self.batch_size
        ), "Dataset must have sufficient samples for training"
        assert (
            len(self.valid_ds) >= self.batch_size
        ), f"Validation dataset must have sufficient number of samples (currently \
            {len(self.valid_ds)}) for training"

        # DataLoader setup
        drop_last = cfg.get("drop_last", True)
        num_workers = cfg.get("num_workers")
        self.dl = get_dataloader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        self.valid_dl = get_dataloader(
            self.valid_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=1
        )

        # Learning rate and optimizer setup
        self.lr = cfg.get("learning_rate")
        self.initial_lr = cfg.get("initial_learning_rate")

        self.optim = get_optimizer(
            model.parameters(),
            lr=cfg.get("learning_rate"),
            wd=cfg.get("wd"),
            betas=cfg.get("betas"),
        )

        self.loss = get_loss(self.project)
        self.accuracy = get_accuracy(self.project)

        # Scheduler setup
        num_train_steps = self.epochs * self.ds.__len__() // self.batch_size
        self.scheduler = CosineAnnealingLR(self.optim, T_max=num_train_steps)

        # Prepare models and optimizers with accelerator
        (
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl,
        )

        # Initialize trackers
        hps = {
            "num_train_steps": num_train_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "learning_rate": self.lr,
            "initial_learning_rate": self.initial_lr,
            "epochs": self.epochs,
        }
        self.accelerator.init_trackers(self.project, config=hps)

        # Initialize best validation metric
        self.best_loss = float("inf")
        self.plot_gt_once = False

    def save(self, path, best_loss):
        """
        Save the current state of the model, optimizers, and schedulers.

        Parameters
        ----------
        path : str
            Path to save the checkpoint.
        best_loss : float
            Current best validation loss to compare and save if improved.
        """
        if best_loss < self.best_loss:
            self.best_loss = best_loss
            torch.save(
                self.accelerator.get_state_dict(self.model),
                f"{self.results_folder}/{self.project}_best_dev.pt",
            )
        ckpts = sorted(Path(path).parent.glob(f"EmbeddingTrainer_*"))
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[: -self.num_ckpt_keep]]
        pkg = dict(
            model=self.accelerator.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            scheduler=self.scheduler.state_dict(),
            best_loss=self.best_loss,
        )
        torch.save(pkg, path)

    def load(self, path=None, restore_optimizer=True):
        """
        Load the model, optimizers, and schedulers from a checkpoint.

        Parameters
        ----------
        path : str, optional
            Path to the checkpoint to load (default is None, which loads the latest checkpoint).
        restore_optimizer : bool, optional
            Whether to restore the optimizer and scheduler states (default is True).
        """
        if not exists(path):
            ckpts = sorted(self.results_folder.glob(f"EmbeddingTrainer_*"))
            path = str(ckpts[-1])
        model = self.accelerator.unwrap_model(self.model)
        pkg = torch.load(path, map_location="cpu")
        model.load_state_dict(pkg["model"])

        if restore_optimizer:
            self.optim.load_state_dict(pkg["optim"])
            self.scheduler.load_state_dict(pkg["scheduler"])
            if "best_loss" in pkg.keys():
                self.best_loss = pkg["best_loss"]
                if self.is_main:
                    self.print(f"The best dev loss before is {self.best_loss}")

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor(
                [checkpoint_num_steps(path) + 1], device=self.device
            )

    def print(self, msg):
        """
        Print a message to the console if in the main process.

        Parameters
        ----------
        msg : str
            The message to print.
        """
        self.accelerator.print(msg)

    @property
    def device(self):
        """
        Device property.

        Returns
        -------
        torch.device
            The device used for training (CPU or GPU).
        """
        return self.accelerator.device

    @property
    def is_distributed(self):
        """
        Check if training is distributed.

        Returns
        -------
        bool
            True if distributed training is enabled, False otherwise.
        """
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        """
        Check if the current process is the main process.

        Returns
        -------
        bool
            True if the process is the main process, False otherwise.
        """
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        """
        Check if the current process is the main process on the local machine.

        Returns
        -------
        bool
            True if the process is the main process on the local machine, False otherwise.
        """
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        """
        Calculate the learning rate for a given training step during warmup.

        Parameters
        ----------
        step : int
            The current training step.

        Returns
        -------
        float
            The learning rate for the current step.
        """
        if step < self.num_warmup_steps:
            return (
                self.initial_lr
                + (self.lr - self.initial_lr) * step / self.num_warmup_steps
            )
        else:
            return self.lr

    def log(self, values: dict, step, type=None, **kwargs):
        """
        Log values to TensorBoard.

        Parameters
        ----------
        values : dict
            Dictionary of values to log.
        step : int
            The current training step.
        type : str, optional
            Type of data to log ('figure', 'audio', or None) (default is None).
        **kwargs : additional keyword arguments
            Additional arguments for TensorBoard logging.
        """
        if type == "figure":
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == "audio":
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)

    def train(self):
        """
        Train the model.

        This function performs training over epochs, updates model parameters,
        and logs training metrics. It also performs validation and saves model checkpoints.
        """
        self.model.train()
        step_time_log = {}

        steps = int(self.steps.item())
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
        else:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

        for epoch in range(self.epochs):
            if self.is_main:
                print(f"Epoch:{epoch} start...")

            for batch in self.dl:
                tic = time.time()

                # Training step
                _, x, y = batch
                x = x.unsqueeze(1)
                y_hat = self.model(x).squeeze()

                # Embedding training
                self.optim.zero_grad()
                loss = self.loss(y_hat, y)
                self.accelerator.backward(loss)
                self.optim.step()

                # Log training metrics
                step_time_log = accum_log(
                    step_time_log, {"time_cost": time.time() - tic}
                )
                if self.is_main and not (steps % self.stdout_steps):
                    with torch.inference_mode():
                        accuracy = self.accuracy(y_hat, y)
                    self.print(
                        f"Epoch {epoch} -- Step {steps}: Loss: {loss.mean():0.3f}; \
                            Accuracy:{accuracy.item():0.3f}; \
                            Time cost per step: {step_time_log['time_cost']:0.3f}s"
                    )
                    step_time_log = {}
                if self.is_main and not (steps % self.log_steps):
                    self.log(
                        {
                            "train/loss": loss.mean(),
                            "train/accuracy": accuracy.mean(),
                            "train/learning_rate": lr,
                        },
                        step=steps,
                    )

                self.accelerator.wait_for_everyone()

                # Validate and save model
                if self.is_main and not (steps % self.save_model_steps) and steps != 0:
                    self.print("Validation start ...")
                    total_loss = 0.0
                    total_accuracy = 0.0
                    num = 0
                    self.model.eval()
                    with torch.inference_mode():
                        for i, batch in tqdm(enumerate(self.valid_dl)):
                            _, x, y = batch
                            x = x.unsqueeze(1)
                            y_hat = self.model(x)
                            y_hat = y_hat.reshape_as(y)
                            loss = self.loss(y_hat, y).item()
                            total_loss += loss
                            accuracy = self.accuracy(y_hat, y)
                            total_accuracy += accuracy
                            num += x.size(0)
                            if i < self.showpiece_num:
                                if not self.plot_gt_once:
                                    self.log(
                                        {f"groundtruth/x_{i}": y.cpu().detach()},
                                        # type="audio",
                                        # sample_rate=self.sample_rate,
                                        step=steps,
                                    )

                                self.log(
                                    {f"generate/y_hat_{i}": y_hat.cpu().detach()},
                                    # type="audio",
                                    # sample_rate=self.sample_rate,
                                    step=steps,
                                )
                        if not self.plot_gt_once:
                            self.plot_gt_once = True
                        self.print(
                            f"{steps}: dev loss: {total_loss / num:0.3f}\tdev accuracy: {total_accuracy / num:0.3f}"
                        )
                        self.log(
                            {
                                "dev/accuracy": total_accuracy / num,
                                "dev/loss": total_loss / num,
                            },
                            step=steps,
                        )

                    # Save model checkpoint
                    model_path = str(
                        self.results_folder / f"EmbeddingTrainer_{steps:08d}"
                    )
                    self.save(model_path, total_accuracy / num)
                    self.print(f"{steps}: saving model to {str(self.results_folder)}")
                    self.model.train()

                # Update learning rate
                self.steps += 1
                steps = int(self.steps.item())
                if steps < self.num_warmup_steps:
                    lr = self.warmup(steps)
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = lr
                else:
                    self.scheduler.step()
                    lr = self.scheduler.get_last_lr()[0]

        self.print("training complete")

    def continue_train(self):
        """
        Continue training from the last checkpoint.

        This method loads the most recent checkpoint and resumes training from that point.
        """
        self.load()
        self.train()

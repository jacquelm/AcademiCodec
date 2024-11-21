from pathlib import Path
import numpy as np
import os
import sys
import json
import re
from beartype import beartype
import joblib

import torch
from sklearn.cluster import MiniBatchKMeans
from .dataset import get_dataloader, EmbeddingDataset

from tqdm import tqdm
from accelerate import (
    Accelerator,
    DistributedType,
    DistributedDataParallelKwargs,
    DataLoaderConfiguration,
)


class KmeansTrainer:
    """
    Trainer class for the Kmeans model, handling training, validation, saving, and
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

        self.n_clusters = cfg.get("km_clusters")
        self.seed = cfg.get("km_seed")
        self.percent = cfg.get("km_percent")
        self.init = cfg.get("km_init")
        self.max_iter = cfg.get("km_iter")
        self.batch_size = cfg.get("km_batch")
        self.tol = cfg.get("km_tol")
        self.max_no_improvement = cfg.get("km_max")
        self.n_init = cfg.get("km_n_init")
        self.reassignment_ratio = cfg.get("km_ratio")
        self.threshold = cfg.get("km_threshold")

        self.km_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            verbose=1,
            compute_labels=False,
            tol=self.tol,
            max_no_improvement=self.max_no_improvement,
            init_size=None,
            n_init=self.n_init,
            reassignment_ratio=self.reassignment_ratio,
        )

        self.model = model

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

        config = cfg.get("embedding_model_config")
        with open(config) as f:
            config = json.load(f)

        self.n_mfcc = config.get("n_mfcc")
        self.n_mel = config.get("n_mel")
        self.n_freq = config.get("n_freq")
        self.n_fft = config.get("n_fft")
        self.hop_size = config.get("hop_size")
        self.win_size = config.get("win_size")
        self.calib = config.get("calib")
        self.volt_max = config.get("volt_max")
        self.win = config.get("win")
        self.step = config.get("step")

        self.audio_type = config.get("audio_type")
        self.showpiece_num = config.get("showpiece_num", 8)
        self.project = config.get("project")
        self.create = False
        self.feature = config.get("feature")

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

        # Load dataset files
        self.csv_dir = config.get("csv_dir", None)
        segment_size = config.get("segment_size")
        train_files = os.path.join(sys.path[0], cfg.get("train_files"))
        with open(train_files, "r") as f:
            train_file_list = f.readlines()

        self.train_ds = EmbeddingDataset(
            file_list=train_file_list,
            segment_size=segment_size,
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mel=self.n_mel,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            calib=self.calib,
            volt_max=self.volt_max,
            win=self.win,
            step=self.step,
            project=self.project,
            create=False,
            feature=self.feature,
            csv_dir=self.csv_dir,
        )
        drop_last = cfg.get("drop_last", True)
        num_workers = cfg.get("num_workers")
        self.train_dl = get_dataloader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        # Prepare models and optimizers with accelerator
        (
            self.model,
            self.km_model,
            self.train_dl,
        ) = self.accelerator.prepare(
            self.model,
            self.km_model,
            self.train_dl,
        )

    def load_feature(self):
        features = torch.zeros((0))
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(self.train_dl)):
                _, x, _ = batch
                x = x.unsqueeze(1)
                feature = self.model.forward_lstm(x)
                feature = self.model.activation_function(self.model.fc1(feature))
                features = torch.cat((features, feature.detach().cpu()))
        return features

    def train(self):
        feat = self.load_feature()
        nonempty = feat.abs().sum(dim=0) > self.threshold
        feat = feat[:, nonempty]
        # feat = feat.T  # (dim, time)
        self.km_model.fit(feat[:, feat.abs().sum(dim=0).bool()])
        joblib.dump(self.km_model, os.path.join(self.results_folder, "kmeans.bin"))
        joblib.dump(nonempty, os.path.join(self.results_folder, "nonempty.bin"))

        inertia = -self.km_model.score(feat) / len(feat)
        print("total intertia: %.5f", inertia)
        print("finished successfully")


class ApplyKmeans(object):
    def __init__(self, km_path, nonempty_path):
        self.km_model = joblib.load(km_path)
        self.nonempty = joblib.load(nonempty_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).float()
        self.Cnorm = torch.from_numpy(self.Cnorm_np).float()
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import argparse

if os.path.exists("/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"
    )
elif os.path.exists("/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec"):
    sys.path.insert(0, "/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec")
else:
    sys.path.insert(0, "/home/mjacquelin/Project/Neural_Network/Pytorch/AcademiCodec")

from academicodec.models.embedding import CNN_LSTM, SPLPredictorCNN, KmeansTrainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="Continue to train from checkpoints",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    embedder = None
    if cfg.get("embedding_model_config") and cfg.get("project") == "classifier":
        embedder = CNN_LSTM.load_from_checkpoint(
            cfg.get("embedding_model_config"), cfg.get("embedding_model_path")
        )
        embedder.eval()
    elif cfg.get("embedding_model_config") and cfg.get("project") == "predictor":
        embedder = SPLPredictorCNN.load_from_checkpoint(
            cfg.get("embedding_model_config"), cfg.get("embedding_model_path")
        )
        embedder.eval()

    trainer = KmeansTrainer(
        model=embedder,
        cfg=cfg,
    )

    trainer.train()

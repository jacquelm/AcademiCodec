import os, sys
import json
import argparse

if os.path.exists("/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"
    )
elif os.path.exists("/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec"):
    sys.path.insert(0, "/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec")
else:
    sys.path.insert(0, "/home/mjacquelin/Project/Neural_Network/Pytorch/AcademiCodec")

from academicodec.models.embedding.d_vector import CNN_LSTM
from academicodec.models.embedding.predictor import SPLPredictorCNN
from academicodec.models.embedding.trainer import EmbeddingTrainer


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

    # Initialize the model
    project = cfg.get("project")
    if project == "classifier":
        model = CNN_LSTM(cfg)
    elif project == "predictor":
        model = SPLPredictorCNN(cfg)
    else:
        raise Exception(f"Incorect name for project : {project}")
    print(model)

    trainer = EmbeddingTrainer(model, cfg=cfg)

    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()

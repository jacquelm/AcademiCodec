import os, sys
import json
from datetime import datetime
import argparse

if os.path.exists("/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"
    )
elif os.path.exists("/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec"):
    sys.path.insert(0, "/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec")
elif os.path.exists("/localdata_ssd/jacquelm/AcademiCodec"):
    sys.path.insert(0, "/localdata_ssd/jacquelm/AcademiCodec")
else:
    sys.path.insert(0, "/home/mjacquelin/Project/Neural_Network/Pytorch/AcademiCodec")

from academicodec.models.speechtokenizer.model import SpeechTokenizer
from academicodec.models.speechtokenizer.trainer import SpeechTokenizerTrainer
from academicodec.modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from academicodec.models.embedding.predictor import SPLPredictorCNN

if __name__ == "__main__":
    print("Start Process", datetime.now())
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
    print("Create Model", datetime.now())
    generator = SpeechTokenizer(cfg)
    discriminators = {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(32),
    }

    embedder = None
    if cfg.get("embedding_model_config"):
        embedder = SPLPredictorCNN.load_from_checkpoint(
            cfg.get("embedding_model_config"), cfg.get("embedding_model_path")
        )
        embedder.eval()
    print("Create Trainer", datetime.now())
    trainer = SpeechTokenizerTrainer(
        generator=generator, discriminators=discriminators, cfg=cfg, embedder=embedder
    )
    print("Start Training", datetime.now())
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
    print("End Training", datetime.now())

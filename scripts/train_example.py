from academicodec.models.speechtokenizer.model import (
    SpeechTokenizer,
)
from academicodec.models.speechtokenizer.trainer import SpeechTokenizerTrainer
from academicodec.models.speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
import json
import argparse


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

    generator = SpeechTokenizer(cfg)
    discriminators = {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(32),
    }

    trainer = SpeechTokenizerTrainer(
        generator=generator, discriminators=discriminators, cfg=cfg
    )

    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()

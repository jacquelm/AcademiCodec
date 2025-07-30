import os, sys, glob
import argparse
from tqdm import tqdm
import numpy as np
import torchaudio
import torch

if os.path.exists("/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Neural_Network/Pytorch/AcademiCodec"
    )
elif os.path.exists("/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec"):
    sys.path.insert(0, "/lustre/fswork/projects/rech/ztm/ulg45lz/AcademiCodec")
else:
    sys.path.insert(0, "/home/mjacquelin/Project/Neural_Network/Pytorch/AcademiCodec")
from academicodec.models.speechtokenizer.model import SpeechTokenizer
from academicodec.utils import find_nearest


def get_parser():
    parser = argparse.ArgumentParser(
        description="Create embeding filesfrom output of encoder"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/speechtokenizer/config_lstm.json",
        help="Config file path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/SpeechTokenizer.pt",
        help="The folder with the weights of the model",
    )
    parser.add_argument(
        "--inputdir",
        type=str,
        default="/data/Audio_tmp/Raw/LibriSpeech/train-clean-100/wavs16/",
        help="The folder with the audio files",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default="/data/Audio_tmp/Raw/LibriSpeech/train-clean-100/",
        help="The output folder with the embedding files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to run the script",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="flac",
        help="The extension of the audio files",
    )
    return parser


def main(args):
    model = SpeechTokenizer.load_from_checkpoint(
        args.config, args.checkpoint_path, map_location=args.device
    )
    model.eval().to(args.device)

    wav_paths = glob.glob(f"{args.inputdir}/*.{args.ext}")
    print(f"Globbed {len(wav_paths)} wav files.")

    emb_tmp = torch.zeros((0))

    # emb_tmp = torch.zeros((0))
    emb_tmp = []

    # os.makedirs(outputdir, exist_ok=True)
    for wav_path in tqdm(wav_paths[::6]):
        orig, sr = torchaudio.load(wav_path)

        # monophonic checking
        if orig.shape[0] > 1:
            orig = orig[:1, :]

        if sr != model.sample_rate:
            orig = torchaudio.functional.resample(orig, sr, model.sample_rate)

        orig = orig.unsqueeze(0).to(args.device)

        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            emb = model.encoder(orig).to("cpu")  # quantized features
        emb_tmp.append(emb)
    emb_tmp = torch.cat(emb_tmp, dim=-1)
    torch.save(emb_tmp, os.path.join(args.outputdir, "embeddings.pt"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", type=str, help="Input folder")
    parser.add_argument("--outputdir", type=str, help="Output folder")
    parser.add_argument("--train_files", type=str, help="Training file")
    parser.add_argument("--valid_files", type=str, help="Validation file")
    parser.add_argument("--ext", type=str, help="Extension files")
    parser.add_argument("--valid", type=float, help="Validation size")
    args = parser.parse_args()

    ext = args.ext
    path = Path(args.inputdir)
    file_list = [str(file) for file in path.glob(f"**/*.{ext}")]
    if args.valid != 0 and args.valid < 1:
        valid_set_size = int(len(file_list) * args.valid)
    else:
        valid_set_size = int(args.valid)
    train_file_list = args.train_files
    valid_file_list = args.valid_files
    if os.path.exists(train_file_list):
        os.remove(train_file_list)
    if valid_file_list and os.path.exists(valid_file_list):
        os.remove(valid_file_list)
    random.shuffle(file_list)
    print(
        f"A total of {len(file_list)} samples will be processed, and {valid_set_size} \
            of them will be included in the validation set."
    )
    file_list = sorted(file_list)
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):

            rep_file = (
                audio_file.replace(args.inputdir, args.outputdir).split(".")[0] + ".npy"
            )
            rep_sub_dir = "/".join(rep_file.split("/")[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            if i < valid_set_size:
                with open(valid_file_list, "a+") as f:
                    f.write(f"{audio_file}\t{rep_file}\n")
            else:
                with open(train_file_list, "a+") as f:
                    f.write(f"{audio_file}\t{rep_file}\n")

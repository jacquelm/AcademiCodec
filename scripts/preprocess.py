import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import resampy
import soundfile as sf
import librosa
from tqdm import tqdm


def get_parser():
    """Get parser for input arguments

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--down", action="store_true")
    parser.add_argument("--postfix", type=str, default="wav")  # wave file extension

    return parser


def pad_data(p, out_dir, trim=False, pad=False, down=False):
    """Preprocessing.

    - Resampling
    - Silence trimming
    - Padding
    """
    # Read
    data, sr = sf.read(p)

    # Resampling
    if sr != 16000 and down:
        data = resampy.resample(data, sr, 16000)
        sr = 16000

    # Silence trimming at Head/Tail
    if trim:
        data, _ = librosa.effects.trim(data, top_db=20)

    # Pad tail for round wave
    if pad:
        unit_length = 1280
        if data.shape[0] % unit_length != 0:
            data = np.pad(
                data,
                (0, unit_length - data.shape[0] % unit_length),
                mode="constant",
                constant_values=0,
            )
        assert data.shape[0] % unit_length == 0

    # Write
    outpath = out_dir / p.name
    outpath.parent.mkdir(exist_ok=True, parents=True)
    sf.write(outpath, data, sr)


def main(args):

    files = list(Path(args.srcdir).glob(f"**/*{args.postfix}"))
    out_dir = Path(args.outdir)
    for file in tqdm(files):
        pad_data(file, out_dir=out_dir, trim=args.trim, pad=args.pad, down=args.down)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

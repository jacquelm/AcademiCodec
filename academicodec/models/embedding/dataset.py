import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from academicodec.modules.embedding import (
    extract_features,
    extract_spectrogram,
    extract_melspectrogram,
    extract_mfcc,
    intensity_level,
    audio_class,
)


# Dataset class for loading audio and corresponding features
class EmbeddingDataset(Dataset):
    """
    Custom dataset class for handling audio data and its corresponding features.
    """

    def __init__(
        self,
        file_list,
        segment_size,
        sample_rate,
        valid=False,
        audio_type=["soft", "normal", "loud", "veryloud"],
        n_mfcc=12,
        n_mel=80,
        n_fft=1024,
        hop_size=240,
        win_size=1024,
        db=10,
        calib=0.1,
        volt_max=1.0,
        win=20,
        step=5,
        project="classifier",
        create=False,
        feature="mfcc",
        additionnal=False,
        csv_dir=None,
    ):
        super().__init__()
        self.file_list = (
            file_list  # List of files with paths to audio and feature files
        )
        self.segment_size = (
            segment_size  # Size of the segment to be extracted from audio
        )
        self.sample_rate = sample_rate  # Desired sample rate for audio
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.db = db

        self.valid = valid  # Flag indicating whether the dataset is for validation

        if project == "classifier":
            self.function_target = lambda a, f: audio_class(
                a, f, self.sample_rate, audio_type
            )
        elif project == "predictor":
            self.function_target = lambda a, f: intensity_level(
                a, f, self.sample_rate, calib, volt_max, win, step
            )

        self.create = create
        self.csv_dir = csv_dir
        self.additionnal = additionnal

        if feature == "spectrogram":
            self.extract_features = lambda a: extract_spectrogram(
                a,
                self.sample_rate,
                self.n_fft,
                self.hop_size,
                self.win_size,
            )
        elif feature == "mel":
            self.extract_features = lambda a: extract_melspectrogram(
                a,
                self.sample_rate,
                self.n_fft,
                self.hop_size,
                self.win_size,
            )
        elif feature == "mfcc":
            self.extract_features = lambda a: extract_mfcc(
                a,
                self.sample_rate,
                self.n_fft,
                self.hop_size,
                self.win_size,
                self.n_mfcc,
            )
        else:
            self.extract_features = lambda a: extract_features(
                a,
                self.sample_rate,
                self.n_fft,
                self.hop_size,
                self.win_size,
                self.n_mfcc,
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Read the file paths from the file_list at the given index
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split("\t")
        speaker = os.path.basename(audio_file).rsplit("_")[0]

        # Load audio file
        audio, sr = torchaudio.load(audio_file)

        # Convert audio to mono
        if len(audio.shape) > 1:
            audio = audio[0, :]
        # Resample audio if sample rate does not match the desired rate
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Remove leading and trailing silence
        audio, _ = librosa.effects.trim(
            audio,
            top_db=self.db,
            frame_length=self.n_fft,
            hop_length=self.hop_size,
        )

        target = self.function_target(audio, audio_file)
        if self.csv_dir:
            df = pd.read_csv(os.path.join(self.csv_dir, f"ltas_features_{speaker}.csv"))
            target_max = df["mean_intensity"].max()
            target_min = df["mean_intensity"].min()
            target_mean = df["mean_intensity"].mean()
            target_std = df["mean_intensity"].std()
            # print("target", target, "max", target_max, "min", target_min)
            target = (target - target_min) / (target_max - target_min)
            target = np.float32(np.maximum(np.minimum(target, 1.0), 0.0))

        audio = audio / audio.abs().max()

        # Handle cases where audio length exceeds segment size
        if audio.size(-1) > self.segment_size:
            if self.valid:
                # For validation, just take the first segment
                audio = audio[: self.segment_size]
                # Extract features if not precomputed
                features = self.extract_features(
                    audio.numpy(),
                )
                if self.additionnal:
                    addtionnal = np.concatenate(
                        (
                            np.ones((1, features.shape[-1])) * target_mean,
                            np.ones((1, features.shape[-1])) * target_std,
                        ),
                        axis=0,
                        dtype=np.float32,
                    )
                    features = np.concatenate((features, addtionnal), axis=0)
                return audio, features, target
            # For training, randomly select a segment
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_size]
        else:
            # if not self.valid:
            # For training, pad the audio if it's shorter than the segment size
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_size - audio.size(-1)), "constant"
            )
        if not os.path.exists(feature_file) or self.create:
            features = self.extract_features(
                audio.numpy(),  # Convert tensor to numpy for librosa compatibility
            )
            if self.additionnal:
                addtionnal = np.concatenate(
                    (
                        np.ones((1, features.shape[-1])) * target_mean,
                        np.ones((1, features.shape[-1])) * target_std,
                    ),
                    axis=0,
                    dtype=np.float32,
                )
                features = np.concatenate((features, addtionnal), axis=0)
            np.save(feature_file, features)

        # Load precomputed features
        features = torch.from_numpy(np.load(feature_file, allow_pickle=True))

        return audio, features, target


def collate_fn(data):
    # Check if the batch data is a list of single tensors or tuples of tensors
    is_one_data = not isinstance(data[0], tuple)

    outputs = []  # List to hold processed tensors
    if is_one_data:
        # Process a batch of single tensors
        for datum in data:
            if isinstance(datum, torch.Tensor):
                # If datum is a tensor, add a new dimension for batch size
                output = datum.unsqueeze(0)
            else:
                # If datum is not a tensor, convert it to a tensor with a batch dimension
                output = torch.tensor([datum])
            outputs.append(output)
        # Return as a tuple of tensors
        return tuple(outputs)

    # Process a batch of tuples of tensors
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            # If elements of datum are tensors, pad them to the same length
            output = pad_sequence(datum, batch_first=True)
        else:
            # If elements are not tensors, convert the list to a tensor
            output = torch.tensor(np.array(datum))
        outputs.append(output)

    # Return as a tuple of tensors
    return tuple(outputs)


def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)

import os
import glob
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from academicodec.modules.torch_embedding import (
    extract_spectrogram,
    extract_melspectrogram,
    extract_mfcc,
    extract_features,
)


class SpeechTokenizerDataset(Dataset):
    """Dataset to load SpeechTokenizer data."""

    def __init__(self, audio_dir, args):
        super().__init__()
        self.audio_dir = audio_dir
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        self.sr = args.sr
        self.max_len = args.audio_duration * args.sr
        self.do_distillation = args.do_distillation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio, sr = torchaudio.load(self.filenames[index])
        resampler = torchaudio.transforms.Resample(sr, self.sr)
        audio = resampler(audio)
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            x_wav = audio[:, st:ed]
        else:
            ans[:, : audio.shape[1]] = audio
            x_wav = ans
        if self.do_distillation:
            filename = os.path.basename(self.filenames[index])
            filename = os.path.splitext(filename)[0]
            teacher_feat_path = f"{self.audio_dir}/teacher_feats/{filename}.pt"
            teacher_unit_path = f"{self.audio_dir}/teacher_units/{filename}.pt"
            teacher_feat = torch.load(teacher_feat_path)
            teacher_unit = torch.load(teacher_unit_path)
            return x_wav, teacher_feat, teacher_unit
        else:
            return x_wav


class audioDataset(Dataset):
    """
    Custom dataset class for handling audio data and its corresponding features.

    Parameters
    ----------
    file_list : list of str
        List of file paths where each path contains the audio file and feature file.
    segment_size : int
        Size of the audio segment to return from the dataset.
    sample_rate : int
        Target sample rate for resampling audio.
    downsample_rate : int, optional
        Downsample rate for feature extraction. Default is 320.
    valid : bool, optional
        Indicator of whether the dataset is used for validation. Default is False.
    """

    def __init__(
        self,
        file_list,
        segment_size,
        sample_rate,
        downsample_rate=320,
        n_mfcc=12,
        n_mel=80,
        n_fft=1024,
        hop_size=240,
        win_size=1024,
        valid=False,
        feature=None,
    ):
        super().__init__()
        self.file_list = (
            file_list  # List of files with paths to audio and feature files
        )
        self.segment_size = (
            segment_size  # Size of the segment to be extracted from audio
        )
        self.sample_rate = sample_rate  # Desired sample rate for audio
        self.valid = valid  # Flag indicating whether the dataset is for validation
        self.downsample_rate = downsample_rate  # Rate for downsampling features

        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.extract_features = None
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
        elif feature == "all":
            self.extract_features = lambda a: extract_features(
                a,
                self.sample_rate,
                self.n_fft,
                self.hop_size,
                self.win_size,
                self.n_mfcc,
            )

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns
        -------
        int
            Number of items in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing the audio tensor and feature tensor.
        """
        # Read the file paths from the file_list at the given index
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split("\t")

        # Load audio file and its corresponding feature file
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file))

        # Convert audio to mono by averaging across channels
        audio = audio.mean(axis=0)

        # Resample audio if sample rate does not match the desired rate
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Handle cases where audio length exceeds segment size
        vocal_feature = None
        if audio.size(-1) > self.segment_size:
            if self.valid:
                if self.extract_features is not None:
                    vocal_feature = self.extract_features(audio[: self.segment_size])
                # For validation, just take the first segment
                return (
                    audio[: self.segment_size],
                    feature[: self.segment_size // self.downsample_rate],
                    vocal_feature,
                )
            # For training, randomly select a segment
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_size]

            # Calculate feature segment start position based on audio segment start
            feature_start = min(
                int(audio_start / self.downsample_rate),
                feature.size(0) - self.segment_size // self.downsample_rate,
            )
            feature = feature[
                feature_start : feature_start
                + self.segment_size // self.downsample_rate,
                :,
            ]
        else:
            # For training, pad the audio if it's shorter than the segment size
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_size - audio.size(-1)), "constant"
            )
        if self.extract_features is not None:
            vocal_feature = self.extract_features(audio)

        return audio, feature, vocal_feature


def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)


# def collate_fn(batch):
#     x_wavs = []
#     teacher_conts = []
#     teacher_units = []

#     for item in batch:
#         if isinstance(item, tuple):
#             x_wav, teacher_cont, teacher_unit = item
#             x_wavs.append(x_wav)
#             teacher_conts.append(teacher_cont)
#             teacher_units.append(teacher_unit)
#         else:
#             x_wavs.append(x_wav)

#     x_wavs = torch.stack(x_wavs)
#     if len(teacher_conts) > 0:
#         teacher_conts = torch.stack(teacher_conts)
#     if len(teacher_units) > 0:
#         teacher_units = torch.stack(teacher_units)

#     if len(teacher_conts) > 0 and len(teacher_units) > 0:
#         return x_wavs, teacher_conts, teacher_units
#     else:
#         return x_wavs


def collate_fn(data):
    """
    Collate function to process a batch of data into tensors.

    Parameters
    ----------
    data : list of tuples or list of tensors
        The batch of data where each element is either a tensor or a tuple of tensors.

    Returns
    -------
    tuple of torch.Tensor
        Processed batch where each element is collated into a tensor.
    """
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
            output = torch.tensor(list(datum))
        outputs.append(output)

    # Return as a tuple of tensors
    return tuple(outputs)

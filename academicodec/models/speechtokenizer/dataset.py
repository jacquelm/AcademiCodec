import os
import glob
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SpeechTokenizerDataset(Dataset):
    """Dataset to load SpeechTokenizer data."""

    def __init__(self, audio_dir, args):
        super().__init__()
        self.audio_dir = audio_dir
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/*.wav"))
        print(len(self.filenames))
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

    def __init__(
        self, file_list, segment_size, sample_rate, downsample_rate=320, valid=False
    ):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.valid = valid
        self.downsample_rate = downsample_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split("\t")
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file))
        audio = audio.mean(axis=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.size(-1) > self.segment_size:
            if self.valid:
                return (
                    audio[: self.segment_size],
                    feature[: self.segment_size // self.downsample_rate],
                )
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_size]
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
            if not self.valid:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(-1)), "constant"
                )
        return audio, feature


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
    # return pad_sequence(data, batch_first=True)
    # return pad_sequence(*data)
    is_one_data = not isinstance(data[0], tuple)
    outputs = []
    if is_one_data:
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            output = pad_sequence(datum, batch_first=True)
        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)

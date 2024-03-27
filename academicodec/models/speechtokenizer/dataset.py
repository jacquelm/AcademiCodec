import os
import glob
import random

import torch
import torchaudio
from torch.utils.data import Dataset
from academicodec.models.speechtokenizer.modules.hubert import HubertFeatureReader


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
            ans[:, :audio.shape[1]] = audio
            x_wav = ans
        if self.do_distillation:
            filename = os.path.basename(self.filenames[index])
            filename = os.path.splitext(filename)[0]
            teacher_feat_path = f'{self.audio_dir}/teacher_feats/{filename}.pt'
            teacher_unit_path = f'{self.audio_dir}/teacher_units/{filename}.pt'
            teacher_feat = torch.load(teacher_feat_path)
            teacher_unit = torch.load(teacher_unit_path)
            return x_wav, teacher_feat, teacher_unit
        else:
            return x_wav
        
def collate_fn(batch): 
    x_wavs = []
    teacher_conts = []
    teacher_units = []
    
    for item in batch:
        if isinstance(item, tuple):
            x_wav, teacher_cont, teacher_unit = item
            x_wavs.append(x_wav)
            teacher_conts.append(teacher_cont)
            teacher_units.append(teacher_unit)
        else:
            x_wavs.append(x_wav)
    
    x_wavs = torch.stack(x_wavs)
    if len(teacher_conts) > 0:
        teacher_conts = torch.stack(teacher_conts)
    if len(teacher_units) > 0:
        teacher_units = torch.stack(teacher_units)
    
    if len(teacher_conts) > 0 and len(teacher_units) > 0:
        return x_wavs, teacher_conts, teacher_units
    else: 
        return x_wavs
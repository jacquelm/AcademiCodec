import torch
import argparse
import glob
import fairseq
import joblib
import soundfile as sf
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import random
import os

class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000, use_cuda=True):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = sf.read(path)
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        print('self.task.cfg.sample_rate:', self.task.cfg.sample_rate)
        assert wav.ndim == 1, wav.ndim
        #assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, x, ref_len=None, channel_id=None):
        with torch.no_grad():
            x = x.float()
            if self.use_cuda:
                x = x.cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                ) #res['x']
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
    
class SaveFeatures(nn.Module):
    def __init__(self, audio_dir, hubert_path, kmeans_path, max_len=16000, sr=16000):
        super().__init__()
        self.filenames = []
        self.audio_dir = audio_dir
        self.filenames.extend(glob.glob(audio_dir + '/*.wav'))
        self.filename_len = len(self.filenames)
        print(len(self.filenames))
        self.sr = sr
        self.max_len = max_len
        self.hubert_reader = HubertFeatureReader(hubert_path, layer=9)
        self.kmeans = joblib.load(kmeans_path)
        
    def forward(self):
        for i in range(self.filename_len):
            ans = torch.zeros(1, self.max_len)
            audio, sr = torchaudio.load(self.filenames[i])
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)
            if audio.shape[1] > self.max_len:
                st = random.randint(0, audio.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                x_wav = audio[:, st:ed]
            else:
                ans[:, :audio.shape[1]] = audio
                x_wav = ans
            
            teacher_feat = self.hubert_reader.get_feats(x_wav)
            teacher_feat = teacher_feat.cpu().numpy()
            teacher_unit = self.kmeans.predict(teacher_feat)
            teacher_feat = torch.from_numpy(teacher_feat)
            teacher_unit = torch.from_numpy(teacher_unit)
            filename = os.path.basename(self.filenames[i])
            filename = os.path.splitext(filename)[0]
            print(filename)
            os.makedirs(f'{self.audio_dir}/teacher_feats', exist_ok=True)
            os.makedirs(f'{self.audio_dir}/teacher_units', exist_ok=True)
            torch.save(teacher_feat, f'{self.audio_dir}/teacher_feats/{filename}.pt')
            torch.save(teacher_unit, f'{self.audio_dir}/teacher_units/{filename}.pt')
        print('done!')
  
def extract(args):
    save_features = SaveFeatures(audio_dir=args.audio_dir,
                                 hubert_path=args.hubert_path,
                                 kmeans_path=args.kmeans_path)
    save_features()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default=None, type=str, help='audio dir')
    parser.add_argument('--hubert_path', default=None, type=str, help='hubert_model_path')
    parser.add_argument('--kmeans_path', default=None, type=str, help='kmeans_model_path')

def main():
    args = get_args()
    extract(args)
            
if __name__ == '__main__':
    main()
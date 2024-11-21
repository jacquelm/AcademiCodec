import librosa
import numpy as np
import torch


# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)


def audio_class(audio, file, sr, type):
    audio_type = file.rsplit(".")[0].rsplit("_")[-1].rsplit(" ")[0]
    audio_index = type.index(audio_type)
    audio_class = torch.zeros((len(type)))
    audio_class[audio_index] = 1
    return audio_class


def calculate_intensity(
    audio,
    sr: int,
    calib: float = 0.1,
    volt_max: float = 1.0,
    win: int = 20,
    step: int = 5,
    idb_min: int = 50,
    idb_max: int = 100,
):
    """_summary_

    Args:
        audio (_type_): Audio signal
        sr (int): Sampling rate
        calib (float): Amplitude of the sampled signal that represent 1V
        volt_max (float): Number of Volt per Pascal
        win (int, optional): Number of samples for one frame. Defaults to 20.
        step (int, optional): Number of samples for shift. Defaults to 5.

    Returns:
        _type_: intensity of the signal at the mic level
    """
    tsig = np.arange(0, len(audio)) / sr
    largfen = int(np.round(win / 1000 * sr))
    pas = int(np.round(step / 1000 * sr))
    n = 0
    k = 1
    l = len(audio)

    idb = np.zeros((0))
    t = np.zeros((0))
    while n + largfen < l:
        sig_win = audio[range(n + 1, n + largfen)]
        sp = np.sqrt(2) * sig_win / volt_max / calib
        idb = np.append(
            idb, 10 * np.log10(1 / len(sp) * sum((sp * sp) / (pow(2 * pow(10, -5), 2))))
        )
        t = np.append(t, tsig[n + largfen])
        n = n + pas
    idb = idb[~((idb < idb_min) | (idb_max > idb_max))]
    return idb


def intensity_level(
    audio,
    file,
    sr: int,
    calib: float = 0.1,
    volt_max: float = 1.0,
    win: int = 20,
    step: int = 5,
    idb_min: int = 50,
    idb_max: int = 100,
):

    return np.mean(
        calculate_intensity(audio, sr, calib, volt_max, win, step, idb_min, idb_max)
    )


def extract_spectrogram(audio, sr=16000, n_fft=1024, hop=80, win=400):
    return librosa.core.spectrum._spectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win
    )


def extract_melspectrogram(audio, sr=16000, n_fft=1024, hop=80, win=400):
    return librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win
    )


def extract_mfcc(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, win_length=win
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # return mfcc
    return np.vstack(
        [
            mfcc,
            delta_mfcc,
            delta2_mfcc,
        ]
    )


def extract_spectral_tilt(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):
    S = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop, win_length=win)
    log_spectrum = np.log1p(np.abs(S))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return np.polyfit(freqs, log_spectrum.mean(axis=1), deg=1)


def extract_pitch(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):
    pitches, magnitudes = librosa.core.piptrack(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, fmin=75, fmax=500
    )
    return np.max(pitches, axis=0)


def extract_energy(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):
    return librosa.feature.rms(y=audio, frame_length=win, hop_length=hop)


# Feature extraction function
def extract_features(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):

    # 1. MFCC, Delta, and Delta-Delta MFCC
    mfccs = extract_mfcc(audio, sr, n_fft, hop, win, n_mfcc)

    # 2. Spectral Tilt (via linear regression on the log power spectrum)
    # spectral_tilt = extract_spectral_tilt(audio, sr, n_fft, hop, win)

    # 3. Pitch (F0)
    pitch_values = extract_pitch(audio, sr, n_fft, hop, win)
    pitch_values = pitch_values[np.newaxis, ...]
    # pitch_values = pitch_values[pitch_values > 0]  # Remove non-pitch values

    # 4. Short-term energy
    # energy = extract_energy(audio, hop, win)

    # Stack the features for CNN input
    # feature_combined = np.vstack(
    #     [mfcc, delta_mfcc, delta2_mfcc, pitch_values, energy, spectral_tilt]
    # )
    feature_combined = np.vstack([mfccs, pitch_values])
    feature_combined = mfccs

    return feature_combined

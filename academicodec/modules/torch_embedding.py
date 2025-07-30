import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


# Spectrogram extraction
def extract_spectrogram(audio, sr=16000, n_fft=1024, hop=80, win=400):
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft, hop_length=hop, win_length=win, power=2
    )
    spectrogram = spectrogram_transform(audio)
    return spectrogram


# Mel Spectrogram extraction
def extract_melspectrogram(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mels=128):
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=n_mels
    )
    melspectrogram = mel_transform(audio)
    return melspectrogram


# MFCC extraction
def extract_mfcc(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20, n_mels=128):
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop,
            "win_length": win,
            "n_mels": n_mels,
        },
    )
    mfcc = mfcc_transform(audio)

    # Calculate delta and delta-delta
    delta_transform = T.ComputeDeltas()
    delta_mfcc = delta_transform(mfcc)
    delta2_mfcc = delta_transform(delta_mfcc)

    # Stack MFCC, Delta, and Delta-Delta
    mfcc_combined = torch.cat((mfcc, delta_mfcc, delta2_mfcc), dim=0)
    return mfcc_combined


# Spectral Tilt
def extract_spectral_tilt(audio, sr=16000, n_fft=1024, hop=80, win=400):
    stft_transform = T.Spectrogram(
        n_fft=n_fft, hop_length=hop, win_length=win, power=None
    )
    S = stft_transform(audio)
    log_spectrum = torch.log1p(S.abs())

    # Compute frequencies
    freqs = torch.linspace(0, sr // 2, steps=log_spectrum.size(0))

    # Average log spectrum along time axis and fit a line
    avg_log_spectrum = log_spectrum.mean(dim=1).numpy()  # Convert to numpy for polyfit
    tilt = torch.tensor(
        np.polyfit(freqs.numpy(), avg_log_spectrum, deg=1)
    )  # Linear regression
    return tilt


# Pitch extraction (F0)
def extract_pitch(audio, sr=16000, n_fft=1024, hop=80, win=400):
    # Use torchaudio's pitch detection (YIN algorithm)
    pitch_transform = T.PitchShift(
        sr, n_steps=0
    )  # A placeholder; use torchaudio.functional if needed
    pitch_values = torchaudio.functional.detect_pitch_frequency(audio, sample_rate=sr)
    return pitch_values


# Energy extraction
def extract_energy(audio, win=400, hop=80):
    rms_transform = T.RMS(frame_length=win, hop_length=hop)
    energy = rms_transform(audio)
    return energy


# Feature extraction function
def extract_features(audio, sr=16000, n_fft=1024, hop=80, win=400, n_mfcc=20):
    # 1. MFCC, Delta, and Delta-Delta MFCC
    mfccs = extract_mfcc(audio, sr, n_fft, hop, win, n_mfcc)

    # 2. Spectral Tilt (via linear regression on the log power spectrum)
    spectral_tilt = extract_spectral_tilt(audio, sr, n_fft, hop, win)

    # 3. Pitch (F0)
    pitch_values = extract_pitch(audio, sr, n_fft, hop, win)
    pitch_values = pitch_values.unsqueeze(0)  # Add a batch dimension

    # 4. Short-term Energy
    energy = extract_energy(audio, win, hop)

    # Stack features for CNN input
    feature_combined = torch.cat([mfccs, pitch_values, energy], dim=0)
    return feature_combined

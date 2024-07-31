#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
# and one prediction.
# see original code here:
# https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846

import librosa
import numpy as np
from PIL import Image
from pypesq import pesq as nb_pesq
from pesq import pesq as wb_pesq
from pystoi import stoi
from jiwer import wer, cer


def compute_median(data):
    """Python function takes a 2D NumPy array data as input and computes
    the median along the columns, along with the 95% confidence interval (CI) for each
    column using the interquartile range (IQR) method

    Args:
        data (_type_): A 2D NumPy array where each column represents a set of values.

    Raises:
        NameError: _description_

    Returns:
        _type_: _description_
    """
    median = np.median(data, axis=0)
    q75, q25 = np.quantile(data, [0.75, 0.25], axis=0)
    iqr = q75 - q25
    CI = 1.57 * iqr / np.sqrt(data.shape[0])
    if np.any(np.isnan(data)):
        raise NameError("nan in data")
    return median, CI


def compute_rmse(reference, prediction):
    """The function calculates the Root Mean Squared Error (RMSE) between two arrays
    prediction and reference

    Args:
        prediction (_type_): predictiond values array.
        reference (_type_): Reference (ground truth) values array.

    Returns:
        _type_: Root Mean Squared Error between prediction and reference.
    """
    # scaling, to get minimum nomrlized-rmse
    alpha = np.sum(prediction * reference) / np.sum(prediction**2)
    # prediction_ = np.expand_dims(prediction, axis=1)
    # alpha = np.linalg.lstsq(prediction_, reference, rcond=None)[0][0]
    prediction_scaled = alpha * prediction

    return np.sqrt(np.square(prediction_scaled - reference).mean())


def compute_rmse_db(reference, prediction):
    """The function calculates the Root Mean Squared Error (RMSE) between two arrays
    prediction and reference in decibel (dB) scale

    Args:
        prediction (_type_): predictiond values array.
        reference (_type_): Reference (ground truth) values array.

    Returns:
        _type_: Root Mean Squared Error in decibels between prediction and reference.
    """
    prediction = np.abs(prediction) ** 2
    prediction = 10.0 * np.log10(np.maximum(1e-5, prediction))
    prediction -= 10.0 * np.log10(np.maximum(1e-5, prediction))
    prediction = np.maximum(prediction, prediction.max() - 80.0)

    reference = np.abs(reference) ** 2
    reference = 10.0 * np.log10(np.maximum(1e-5, reference))
    reference -= 10.0 * np.log10(np.maximum(1e-5, reference))
    reference = np.maximum(reference, reference.max() - 80.0)

    return np.sqrt(np.square(reference - prediction).mean())


def check_same_shape(reference, prediction):
    """Check that predictions and reference have the same shape, else raise error."""
    if prediction.shape != reference.shape:
        raise RuntimeError(
            "Inputs are expected to have the same shape"
            + f"but got {prediction.shape} and {reference.shape}."
        )


def compute_snr(reference, prediction, zero_mean: bool = False):
    r"""Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level
    of the desired signal to the level of background noise. Therefore, a high value of
    SNR means that the audio is clear.

    Args:
        prediction: float tensor with shape ``(...,time)``
        reference: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean reference and prediction or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``prediction`` and ``reference`` does not have the same shape

    """
    check_same_shape(reference, prediction)
    eps = np.finfo(prediction.dtype).eps

    if zero_mean:
        reference -= np.mean(reference)
        prediction -= np.mean(prediction)

    noise = reference - prediction

    snr_value = (np.sum(reference**2) + eps) / (np.sum(noise**2) + eps)
    return 10 * np.log10(snr_value)


def compute_sisdr(reference, prediction, zero_mean: bool = False):
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source
    sound.

    Args:
        prediction: float tensor with shape ``(...,time)``
        reference: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean reference and prediction or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``prediction`` and ``reference`` does not have the same shape
    """
    check_same_shape(reference, prediction)
    eps = np.finfo(prediction.dtype).eps

    if zero_mean:
        reference = reference - np.mean(reference)
        prediction = prediction - np.mean(prediction)

    alpha = (np.sum(prediction * reference) + eps) / (np.sum(reference**2) + eps)
    reference_scaled = alpha * reference

    noise = reference_scaled - prediction

    val = (np.sum(reference_scaled**2) + eps) / (np.sum(noise**2) + eps)
    return 10 * np.log10(val)


def compute_lsd(reference, prediction, n_fft: int = 1024, hop_length: int = 256):
    """_summary_

    Args:
        reference (_type_): _description_
        prediction (_type_): _description_
        n_fft (int, optional): _description_. Defaults to 1024.
        hop_length (int, optional): _description_. Defaults to 256.

    Returns:
        _type_: _description_
    """
    eps = np.finfo(prediction.dtype).eps
    reference_sp = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length))
    prediction_sp = np.abs(librosa.stft(prediction, n_fft=n_fft, hop_length=hop_length))
    reference_sp = np.transpose(reference_sp, (1, 0))
    prediction_sp = np.transpose(prediction_sp, (1, 0))
    lsd = np.log10(reference_sp**2 / ((prediction_sp + eps) ** 2) + eps) ** 2
    lsd = np.mean(np.mean(lsd, axis=-1) ** 0.5, axis=0)
    return lsd


def compute_sdr(reference, prediction, scaling=True):
    """The function calculates the Signal-to-Distortion Ratio (SDR),
    Signal-to-Interference Ratio (SIR), Signal-to-Artifacts Ratio (SAR), between two
    arrays in decibel (dB) scale


    Args:
        reference (_type_): _description_
        prediction (_type_): _description_
        scaling (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    a = 1
    reference = np.float32(reference)
    prediction = np.float32(prediction)
    eps = np.finfo(reference.dtype).eps
    reference = reference.reshape((reference.size, 1))
    prediction = prediction.reshape((prediction.size, 1))
    rss = np.dot(reference.T, reference)
    if scaling:
        # get the scaling factor
        a = (eps + np.dot(prediction.T, reference)) / (rss + eps)

    e_true = a * reference
    e_res = prediction - e_true

    sss = (e_true**2).sum()
    snn = (e_res**2).sum()

    sdr = 10 * np.log10((sss + eps) / (snn + eps))

    # Get the SIR and SAR
    rsr = np.dot(reference.T, e_res)
    b = np.linalg.solve(rss, rsr)

    e_interf = np.dot(reference, b)
    e_artif = e_res - e_interf

    sir = 10 * np.log10((sss + eps) / ((e_interf**2).sum() + eps))
    sar = 10 * np.log10((sss + eps) / ((e_artif**2).sum() + eps))

    return sdr, sir, sar


class EvalMetrics:
    # The class is designed for evaluating audio quality metrics between an predictiond
    # signal (prediction) and a reference signal (reference). It supports various metrics
    # such as RMSE, RMSE in dB, SISDR, SISDR from the spectrogram, PESQ, STOI, ESTOI,
    # or a combination of all.
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_size: int = 256,
        metric: str = "all",
        suffix: str = ".wav",
    ):
        self.sr = sr
        self.metric = metric
        self.suffix = suffix
        self.n_fft = n_fft
        self.hop_size = hop_size

    def eval(self, reference, prediction, txt_ref="", txt_pred=""):
        if self.suffix == ".wav" or self.suffix == ".flac":
            # reference, sr_ref = librosa.load(path_ref, sr=16000)
            # prediction, sr = librosa.load(path_pred, sr=16000)

            # mono channel
            if len(prediction.shape) > 1:
                prediction = prediction[:, 0]
            if len(reference.shape) > 1:
                reference = reference[:, 0]

            # # correction temporel
            # prediction = correction_temporel(prediction, reference)
            # # remove silence
            # prediction, _ = librosa.effects.trim(prediction, top_db=30)
            # reference, _ = librosa.effects.trim(reference, top_db=30)
            # align
            len_x = np.min([len(prediction), len(reference)])
            prediction = prediction[:len_x]
            reference = reference[:len_x]
            # center
            prediction = prediction - prediction.mean()
            reference = reference - reference.mean()

            # Spectrum
            spec_est = librosa.stft(prediction, center=True)
            spec_ref = librosa.stft(reference, center=True)

            # reference = reference / np.max(np.abs(reference))

            # if sr != sr_ref:
            #     raise ValueError(
            #         "Sampling rate is different for predictiond audio and reference audio"
            #     )
        elif self.suffix == ".png":
            # load the image
            img_est = Image.open(prediction)
            img_ref = Image.open(reference)

            # convert image to numpy array
            prediction = np.asarray(img_est)
            reference = np.asarray(img_ref)

        elif self.suffix == ".txt":
            pass

        else:
            raise RuntimeError(f"Inputs type {self.suffix} not recognised ")

        if self.metric == "rmse":
            return compute_rmse(reference, prediction)
        elif self.metric == "rmse_db":
            return compute_rmse_db(spec_ref, spec_est)
        elif self.metric == "snr":
            return compute_snr(reference, prediction)
        elif self.metric == "sisdr":
            return compute_sisdr(reference, prediction)
        elif self.metric == "lsd":
            return compute_lsd(reference, prediction, self.n_fft, self.hop_size)
        elif self.metric == "nb_pesq":
            return nb_pesq(reference, prediction, self.sr)
        elif self.metric == "wb_pesq":
            return wb_pesq(self.sr, reference, prediction)
        elif self.metric == "stoi":
            return stoi(reference, prediction, self.sr, extended=False)
        elif self.metric == "estoi":
            return stoi(reference, prediction, self.sr, extended=True)
        elif self.metric == "wer":
            return wer(reference, prediction)
        elif self.metric == "cer":
            return wer(reference, prediction)
        elif self.metric == "all":
            score_rmse = compute_rmse(reference, prediction)
            score_rmse_db = compute_rmse_db(reference, prediction)
            score_snr = compute_snr(reference, prediction)
            score_sisdr = compute_sisdr(reference, prediction)
            score_lsd = compute_lsd(reference, prediction, self.n_fft, self.hop_size)
            score_nbpesq = nb_pesq(reference, prediction, self.sr)
            try:
                wb_pesq(self.sr, reference, prediction)
            except:
                score_wbpesq = np.nan
            else:
                score_wbpesq = wb_pesq(self.sr, reference, prediction)
            score_stoi = stoi(reference, prediction, self.sr, extended=False)
            score_estoi = stoi(reference, prediction, self.sr, extended=True)
            score_wer = wer(txt_ref, txt_pred) if len(txt_ref) > 0 else np.nan
            score_cer = cer(txt_ref, txt_pred) if len(txt_ref) > 0 else np.nan
            return (
                score_rmse,
                score_rmse_db,
                score_snr,
                score_sisdr,
                score_lsd,
                score_nbpesq,
                score_wbpesq,
                score_stoi,
                score_estoi,
                score_wer,
                score_cer,
            )
        else:
            raise ValueError(
                "Evaluation only support: rmse, sdr, lsd, pesq, (e)stoi, wer, all"
            )

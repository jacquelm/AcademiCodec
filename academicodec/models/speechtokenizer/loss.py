import torch
import torchaudio
import matplotlib.pylab as plt


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values to be compressed.
    C : float, optional
        Compression constant. Default is 1.
    clip_val : float, optional
        Minimum value to avoid log of zero. Default is 1e-5.

    Returns
    -------
    torch.Tensor
        Tensor with dynamically compressed range.
    """
    # Clamp the input tensor to avoid log of zero and then apply logarithmic compression
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """
    Normalize the magnitudes of the spectrogram using dynamic range compression.

    Parameters
    ----------
    magnitudes : torch.Tensor
        Input magnitudes of the spectrogram.

    Returns
    -------
    torch.Tensor
        Normalized magnitudes.
    """
    # Apply dynamic range compression to the magnitudes
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}  # Cache for Mel scale basis filters
hann_window = {}  # Cache for Hann window functions


def mel_spectrogram(
    y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute the Mel spectrogram of an audio signal.

    Parameters
    ----------
    y : torch.Tensor
        Input audio waveform. Shape: (batch, channels, timesteps).
    n_fft : int
        Number of FFT bins.
    num_mels : int
        Number of Mel filterbanks.
    sample_rate : int
        Sample rate of the audio signal.
    hop_size : int
        Number of samples between frames.
    win_size : int
        Size of the window used for STFT.
    fmin : float
        Minimum frequency for Mel scale.
    fmax : float
        Maximum frequency for Mel scale.
    center : bool, optional
        Whether to center the window. Default is False.

    Returns
    -------
    torch.Tensor
        Mel spectrogram. Shape: (batch, num_mels, timesteps).
    """
    global mel_basis, hann_window

    # Initialize Mel basis and Hann window if not already cached
    if fmax not in mel_basis:
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=num_mels,  # Number of Mel bins
            sample_rate=sample_rate,  # Sample rate
            n_stft=n_fft // 2 + 1,  # Number of FFT bins
            f_min=fmin,  # Minimum frequency
            f_max=fmax,  # Maximum frequency
            norm="slaney",  # Normalization type
            mel_scale="htk",  # Mel scale type
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = mel_transform.fb.float().T.to(
            y.device  # Store Mel basis on the same device as input
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Pad the input waveform to match the FFT size
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Compute the Short-Time Fourier Transform (STFT)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],  # Hann window function
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.abs(spec) + 1e-9  # Ensure no zero values
    spec = torch.matmul(
        mel_basis[str(fmax) + "_" + str(y.device)], spec
    )  # Apply Mel filter
    spec = spectral_normalize_torch(spec)  # Normalize the spectrogram

    return spec


def plot_spectrogram(spectrogram):
    """
    Plot the spectrogram using matplotlib.

    Parameters
    ----------
    spectrogram : torch.Tensor
        Mel spectrogram to be plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plotted spectrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 2))  # Create a subplot with a specific size
    im = ax.imshow(
        spectrogram, aspect="auto", origin="lower", interpolation="none"
    )  # Display spectrogram
    plt.colorbar(im, ax=ax)  # Add color bar

    fig.canvas.draw()  # Draw the figure
    plt.close()  # Close the plot to avoid display in notebook

    return fig


def recon_loss(x, x_hat):
    """
    Compute the reconstruction loss between original and reconstructed audio.

    Parameters
    ----------
    x : torch.Tensor
        Original audio waveform.
    x_hat : torch.Tensor
        Reconstructed audio waveform.

    Returns
    -------
    torch.Tensor
        L1 loss between original and reconstructed audio.
    """
    length = min(
        x.size(-1), x_hat.size(-1)
    )  # Consider the minimum length to avoid mismatch
    return torch.nn.functional.l1_loss(x[:, :, :length], x_hat[:, :, :length])


def mel_loss(x, x_hat, **kwargs):
    """
    Compute the loss between Mel spectrograms of original and reconstructed audio.

    Parameters
    ----------
    x : torch.Tensor
        Original audio waveform.
    x_hat : torch.Tensor
        Reconstructed audio waveform.
    kwargs : dict
        Additional parameters for mel_spectrogram function.

    Returns
    -------
    torch.Tensor
        L1 loss between Mel spectrograms of original and reconstructed audio.
    """
    # Compute Mel spectrograms for both original and reconstructed audio
    x_mel = mel_spectrogram(x.squeeze(1), **kwargs)
    x_hat_mel = mel_spectrogram(x_hat.squeeze(1), **kwargs)
    length = min(
        x_mel.size(2), x_hat_mel.size(2)
    )  # Consider the minimum length to avoid mismatch
    return torch.nn.functional.l1_loss(x_mel[:, :, :length], x_hat_mel[:, :, :length])


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between real and generated feature maps.

    Parameters
    ----------
    fmap_r : list of torch.Tensor
        List of feature maps from the real data.
    fmap_g : list of torch.Tensor
        List of feature maps from the generated data.

    Returns
    -------
    torch.Tensor
        Loss between real and generated feature maps.
    """
    loss = 0
    # Iterate through real and generated feature maps
    for dr, dg in zip(fmap_r, fmap_g):
        # Compute the loss for each pair of feature maps
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2  # Scale the loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the loss for the discriminator.

    Parameters
    ----------
    disc_real_outputs : list of torch.Tensor
        Discriminator outputs for real data.
    disc_generated_outputs : list of torch.Tensor
        Discriminator outputs for generated data.

    Returns
    -------
    torch.Tensor
        Loss for the discriminator.
    """
    loss = 0
    # Iterate through real and generated discriminator outputs
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)  # Loss for real data
        g_loss = torch.mean(dg**2)  # Loss for generated data
        loss += r_loss + g_loss

    return loss


def adversarial_loss(disc_outputs):
    """
    Compute the adversarial loss for the generator.

    Parameters
    ----------
    disc_outputs : list of torch.Tensor
        Discriminator outputs for generated data.

    Returns
    -------
    torch.Tensor
        Adversarial loss for the generator.
    """
    loss = 0
    # Iterate through discriminator outputs
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)  # Adversarial loss for each output
        loss += l

    return loss


def d_axis_distill_loss(feature, target_feature):
    """
    Compute the distillation loss between features for the D axis.

    Parameters
    ----------
    feature : torch.Tensor
        Features from the student model.
    target_feature : torch.Tensor
        Features from the target model.

    Returns
    -------
    torch.Tensor
        Distillation loss for the D axis.
    """
    n = min(
        feature.size(1), target_feature.size(1)
    )  # Consider the minimum length to avoid mismatch
    # Compute cosine similarity and apply sigmoid function
    distill_loss = -torch.log(
        torch.sigmoid(
            torch.nn.functional.cosine_similarity(
                feature[:, :n], target_feature[:, :n], axis=1
            )
        )
    ).mean()
    return distill_loss


def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    """
    Compute the distillation loss between features for the T axis.

    Parameters
    ----------
    feature : torch.Tensor
        Features from the student model.
    target_feature : torch.Tensor
        Features from the target model.
    lambda_sim : float, optional
        Weight for the similarity loss term. Default is 1.

    Returns
    -------
    torch.Tensor
        Distillation loss for the T axis.
    """
    n = min(
        feature.size(1), target_feature.size(1)
    )  # Consider the minimum length to avoid mismatch
    l1_loss = torch.nn.functional.l1_loss(
        feature[:, :n], target_feature[:, :n], reduction="mean"
    )  # L1 loss
    sim_loss = -torch.log(
        torch.sigmoid(
            torch.nn.functional.cosine_similarity(
                feature[:, :n], target_feature[:, :n], axis=-1
            )
        )
    ).mean()  # Similarity loss
    distill_loss = l1_loss + lambda_sim * sim_loss  # Combine losses
    return distill_loss

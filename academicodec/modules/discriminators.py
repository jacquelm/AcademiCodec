import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import typing as tp
import torchaudio
from einops import rearrange
from . import NormConv2d

LRELU_SLOPE = 0.1


# Function to calculate padding for a convolution operation
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# Function to initialize the weights of a given module
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__  # Get the name of the class
    if classname.find("Conv") != -1:  # Check if the module is a convolution layer
        m.weight.data.normal_(
            mean, std
        )  # Initialize the weights with a normal distribution


# Class defining a Periodic Discriminator
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(
            DiscriminatorP, self
        ).__init__()  # Initialize the base class (torch.nn.Module)
        self.period = period  # Set the period for reshaping input
        norm_f = (
            weight_norm if use_spectral_norm == False else spectral_norm
        )  # Choose normalization function
        self.convs = nn.ModuleList(  # Create a list of convolutional layers
            [
                norm_f(
                    Conv2d(
                        1,  # Number of input channels
                        32,  # Number of output channels
                        (kernel_size, 1),  # Kernel size
                        (stride, 1),  # Stride
                        padding=(get_padding(5, 1), 0),  # Padding for the convolution
                    )
                ),
                norm_f(
                    Conv2d(
                        32,  # Number of input channels
                        128,  # Number of output channels
                        (kernel_size, 1),  # Kernel size
                        (stride, 1),  # Stride
                        padding=(get_padding(5, 1), 0),  # Padding for the convolution
                    )
                ),
                norm_f(
                    Conv2d(
                        128,  # Number of input channels
                        512,  # Number of output channels
                        (kernel_size, 1),  # Kernel size
                        (stride, 1),  # Stride
                        padding=(get_padding(5, 1), 0),  # Padding for the convolution
                    )
                ),
                norm_f(
                    Conv2d(
                        512,  # Number of input channels
                        1024,  # Number of output channels
                        (kernel_size, 1),  # Kernel size
                        (stride, 1),  # Stride
                        padding=(get_padding(5, 1), 0),  # Padding for the convolution
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,  # Number of input channels
                        1024,  # Number of output channels
                        (kernel_size, 1),  # Kernel size
                        1,  # Stride
                        padding=(2, 0),  # Padding for the convolution
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        )  # Final convolution layer

    def forward(self, x):
        fmap = []  # List to store feature maps

        # Reshape input from 1D to 2D based on period
        b, c, t = x.shape  # Get batch size, channels, and time steps
        if (
            t % self.period != 0
        ):  # If time steps are not a multiple of period, pad the input
            n_pad = self.period - (t % self.period)  # Calculate padding size
            x = F.pad(x, (0, n_pad), "reflect")  # Pad the input with reflection padding
            t = t + n_pad  # Update time steps after padding
        x = x.view(b, c, t // self.period, self.period)  # Reshape the input to 2D

        for l in self.convs:  # Apply each convolutional layer
            x = l(x)  # Apply convolution
            x = F.leaky_relu(x, LRELU_SLOPE)  # Apply LeakyReLU activation
            fmap.append(x)  # Store the feature map
        x = self.conv_post(x)  # Apply the final convolution layer
        fmap.append(x)  # Store the final feature map
        x = torch.flatten(x, 1, -1)  # Flatten the output

        return x, fmap  # Return the final output and feature maps


# Class defining a Multi-Period Discriminator
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()  # Initialize the base class
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),  # Create DiscriminatorP with period 2
                DiscriminatorP(3),  # Create DiscriminatorP with period 3
                DiscriminatorP(5),  # Create DiscriminatorP with period 5
                DiscriminatorP(7),  # Create DiscriminatorP with period 7
                DiscriminatorP(11),  # Create DiscriminatorP with period 11
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []  # List to store real outputs
        y_d_gs = []  # List to store generated outputs
        fmap_rs = []  # List to store real feature maps
        fmap_gs = []  # List to store generated feature maps
        for i, d in enumerate(self.discriminators):  # Iterate over all discriminators
            y_d_r, fmap_r = d(y)  # Process real input through discriminator
            y_d_g, fmap_g = d(y_hat)  # Process generated input through discriminator
            y_d_rs.append(y_d_r)  # Store real output
            fmap_rs.append(fmap_r)  # Store real feature map
            y_d_gs.append(y_d_g)  # Store generated output
            fmap_gs.append(fmap_g)  # Store generated feature map

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs  # Return all outputs and feature maps


# Class defining a Single Discriminator
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()  # Initialize the base class
        norm_f = (
            weight_norm if use_spectral_norm == False else spectral_norm
        )  # Choose normalization function
        self.convs = nn.ModuleList(  # Create a list of convolutional layers
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),  # First convolutional layer
                norm_f(
                    Conv1d(128, 128, 41, 2, groups=4, padding=20)
                ),  # Second convolutional layer
                norm_f(
                    Conv1d(128, 256, 41, 2, groups=16, padding=20)
                ),  # Third convolutional layer
                norm_f(
                    Conv1d(256, 512, 41, 4, groups=16, padding=20)
                ),  # Fourth convolutional layer
                norm_f(
                    Conv1d(512, 1024, 41, 4, groups=16, padding=20)
                ),  # Fifth convolutional layer
                norm_f(
                    Conv1d(1024, 1024, 41, 1, groups=16, padding=20)
                ),  # Sixth convolutional layer
                norm_f(
                    Conv1d(1024, 1024, 5, 1, padding=2)
                ),  # Seventh convolutional layer
            ]
        )
        self.conv_post = norm_f(
            Conv1d(1024, 1, 3, 1, padding=1)
        )  # Final convolutional layer

    def forward(self, x):
        fmap = []  # List to store feature maps
        for l in self.convs:  # Apply each convolutional layer
            x = l(x)  # Apply convolution
            x = F.leaky_relu(x, LRELU_SLOPE)  # Apply LeakyReLU activation
            fmap.append(x)  # Store the feature map
        x = self.conv_post(x)  # Apply the final convolution layer
        fmap.append(x)  # Store the final feature map
        x = torch.flatten(x, 1, -1)  # Flatten the output

        return x, fmap  # Return the final output and feature maps


# Class defining a Multi-Scale Discriminator
class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()  # Initialize the base class
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(
                    use_spectral_norm=True
                ),  # Create DiscriminatorS with spectral normalization
                DiscriminatorS(),  # Create DiscriminatorS without spectral normalization
                DiscriminatorS(),  # Create DiscriminatorS without spectral normalization
            ]
        )
        self.meanpools = nn.ModuleList(
            [
                AvgPool1d(4, 2, padding=2),
                AvgPool1d(4, 2, padding=2),
            ]  # Create mean pooling layers
        )

    def forward(self, y, y_hat):
        y_d_rs = []  # List to store real outputs
        y_d_gs = []  # List to store generated outputs
        fmap_rs = []  # List to store real feature maps
        fmap_gs = []  # List to store generated feature maps
        for i, d in enumerate(self.discriminators):  # Iterate over all discriminators
            if i != 0:  # If not the first discriminator
                y = self.meanpools[i - 1](y)  # Downsample real input
                y_hat = self.meanpools[i - 1](y_hat)  # Downsample generated input
            y_d_r, fmap_r = d(y)  # Process real input through discriminator
            y_d_g, fmap_g = d(y_hat)  # Process generated input through discriminator
            y_d_rs.append(y_d_r)  # Store real output
            fmap_rs.append(fmap_r)  # Store real feature map
            y_d_gs.append(y_d_g)  # Store generated output
            fmap_gs.append(fmap_g)  # Store generated feature map

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs  # Return all outputs and feature maps


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        # print('x ', x.shape)
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        # print('z ', z.shape)
        z = torch.cat([z.real, z.imag], dim=1)
        # print('cat_z ', z.shape)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            # print('z i', i, z.shape)
            fmap.append(z)
        z = self.conv_post(z)
        # print('logit ', z.shape)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.List[int] = [1024, 2048, 512, 256, 128],
        hop_lengths: tp.List[int] = [256, 512, 128, 64, 32],
        win_lengths: tp.List[int] = [1024, 2048, 512, 256, 128],
        **kwargs
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        # Create a list of STFT-based discriminators with different FFT, hop,
        # and window sizes
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                    **kwargs
                )
                for i in range(len(n_ffts))
            ]
        )
        self.num_discriminators = len(self.discriminators)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> DiscriminatorOutput:
        logits = []  # List to store real outputs
        logits_fake = []
        fmaps = []  # List to store real feature maps
        fmaps_fake = []
        for disc in self.discriminators:
            logit, fmap = disc(y)
            logits.append(logit)
            fmaps.append(fmap)
            logit_fake, fmap_fake = disc(y_hat)
            logits_fake.append(logit_fake)
            fmaps_fake.append(fmap_fake)
        return logits, logits_fake, fmaps, fmaps_fake

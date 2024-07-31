"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp
import numpy as np
import torch
import torch.nn as nn

from . import (
    SConv1d,
    SConvTranspose1d,
    SLSTM,
    STransformer
)


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params) if activation != 'Snake' else act(in_chs),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder."""
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 bidirectional: bool = False, transformer: int = 2, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        mult = 1
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)
                ]

            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2, kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        if transformer:
            model += [STransformer(mult * n_filters, num_layers=transformer, num_heads=num_heads,
                                   dim_feedforward=dim_feedforward, dropout=dropout)]

        mult = mult * 2 if bidirectional else mult
        model += [
            act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder."""
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 trim_right_ratio: float = 1.0, bidirectional: bool = False, transformer: int = 2, num_heads: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        if transformer:
            model += [STransformer(mult * n_filters, num_layers=transformer, num_heads=num_heads,
                                   dim_feedforward=dim_feedforward, dropout=dropout)]

        for i, ratio in enumerate(self.ratios):
            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params, causal=causal, trim_right_ratio=trim_right_ratio)
            ]
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1], activation=activation,
                                      activation_params=activation_params, norm=norm, norm_params=norm_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)
                ]
            mult //= 2

        model += [
            act(**activation_params) if activation != 'Snake' else act(n_filters),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


def test():
    import torch
    encoder = SEANetEncoder()
    decoder = SEANetDecoder()
    x = torch.randn(1, 1, 24000)
    z = encoder(x)
    print('z', z.shape)
    assert 1==2
    assert list(z.shape) == [1, 128, 75], z.shape
    y = decoder(z)
    assert y.shape == x.shape, (x.shape, y.shape)


if __name__ == '__main__':
    test()

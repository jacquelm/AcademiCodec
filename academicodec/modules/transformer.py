# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A streamable transformer."""
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000):
    """Create time embedding for the given positions, target dimension `dim`."""
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=positions.device).view(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


class StreamingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self, x: torch.Tensor, x_past: torch.Tensor, past_context: int
    ):  # type: ignore
        if self.norm_first:
            sa_input = self.norm1(x)
            x = x + self._sa_block(sa_input, x_past, past_context)
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_input = x
            x = self.norm1(x + self._sa_block(sa_input, x_past, past_context))
            x = self.norm2(x + self._ff_block(x))

        return x, sa_input

    # self-attention block
    def _sa_block(
        self, x: torch.Tensor, x_past: torch.Tensor, past_context: int
    ):  # type: ignore
        _, T, _ = x.shape
        _, H, _ = x_past.shape

        queries = x
        keys = torch.cat([x_past, x], dim=1)
        values = keys

        queries_pos = torch.arange(H, T + H, device=x.device).view(-1, 1)
        keys_pos = torch.arange(T + H, device=x.device).view(1, -1)
        delta = queries_pos - keys_pos
        valid_access = (delta >= 0) & (delta <= past_context)
        x = self.self_attn(
            queries, keys, values, attn_mask=~valid_access, need_weights=False
        )[0]
        return self.dropout1(x)


class StreamingTransformerEncoder(nn.Module):
    """TransformerEncoder with streaming support.

    Args:
        dim (int): dimension of the data.
        hidden_scale (int): intermediate dimension of FF module is this times the dimension.
        num_heads (int): number of heads.
        num_layers (int): number of layers.
        max_period (float): maxium period of cosines in the positional embedding.
        past_context (int or None): receptive field for the causal mask, infinite if None.
        gelu (bool): if true uses GeLUs, otherwise use ReLUs.
        norm_in (bool): normalize the input.
        dropout (float): dropout probability.
        **kwargs: See `nn.TransformerEncoderLayer`.
    """

    def __init__(
        self,
        dim,
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 5,
        max_period: float = 10000,
        past_context: int = 1000,
        gelu: bool = True,
        norm_in: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        assert dim % num_heads == 0
        hidden_dim = int(dim * hidden_scale)

        self.max_period = max_period
        self.past_context = past_context
        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
        else:
            self.norm_in = nn.Identity()

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(
                StreamingTransformerEncoderLayer(
                    dim,
                    num_heads,
                    hidden_dim,
                    activation=activation,
                    batch_first=True,
                    dropout=dropout,
                    **kwargs
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        states: tp.Optional[tp.List[torch.Tensor]] = None,
        offset: tp.Union[int, torch.Tensor] = 0,
    ):
        B, T, C = x.shape
        if states is None:
            states = [torch.zeros_like(x[:, :1]) for _ in range(1 + len(self.layers))]

        positions = torch.arange(T, device=x.device).view(1, -1, 1) + offset
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)

        new_state: tp.List[torch.Tensor] = []
        x = self.norm_in(x)
        x = x + pos_emb

        for layer_state, layer in zip(states, self.layers):
            x, new_layer_state = layer(x, layer_state, self.past_context)
            new_layer_state = torch.cat([layer_state, new_layer_state], dim=1)
            new_state.append(new_layer_state[:, -self.past_context :, :])
        return x, new_state, offset + T


class STransformer(nn.Module):
    """
    Transformer without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        skip: bool = True,
    ):
        super().__init__()
        self.skip = skip
        self.transformer = nn.Transformer(
            d_model=dimension,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.linear = nn.Linear(
            dimension, dimension
        )  # To ensure the dimension match for skip connection

    def forward(self, x):
        """
        Forward pass for the STransformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size, seq_len).
        """
        # Permute to (seq_len, batch_size, input_size)
        x = x.permute(2, 0, 1)

        # Generate mask for the transformer (optional, can be customized)
        seq_len, batch_size, _ = x.size()
        src_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(
            x.device
        )

        # Pass through Transformer (assuming encoder-only usage for simplicity)
        y = self.transformer.encoder(x, mask=src_mask)

        # Apply skip connection if enabled
        if self.skip:
            y = y + self.linear(x)

        # Permute back to (batch_size, input_size, seq_len)
        y = y.permute(1, 2, 0)
        return y

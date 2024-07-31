import torch
from torch import nn

class STransformer(nn.Module):
    """
    Transformer without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.transformer = nn.Transformer(
            d_model=dimension,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.linear = nn.Linear(dimension, dimension)  # To ensure the dimension match for skip connection

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
        src_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through Transformer (assuming encoder-only usage for simplicity)
        y = self.transformer.encoder(x, mask=src_mask)
        
        # Apply skip connection if enabled
        if self.skip:
            y = y + self.linear(x)
        
        # Permute back to (batch_size, input_size, seq_len)
        y = y.permute(1, 2, 0)
        return y

# Example usage:
# dimension = 128
# transformer_layer = STransformer(dimension)
# input_tensor = torch.rand((32, dimension, 100))  # (batch_size, input_size, seq_len)
# output_tensor = transformer_layer(input_tensor)

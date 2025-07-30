import torch.nn as nn
from einops import rearrange, repeat
import torch
import numpy as np

from academicodec.modules import SEANetEncoder, SEANetDecoder
from academicodec.quantization import ResidualVectorQuantizer
from academicodec.quantization.core_vq import StaticResidualVectorQuantizer


class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        """
        Initialize the SpeechTokenizer model.

        Parameters
        ----------
        config : json
            Model configuration parameters.
        """
        super().__init__()

        # Initialize the encoder part of the model with parameters from the config.
        self.encoder = SEANetEncoder(
            n_filters=config.get("n_filters"),  # Number of filters for the encoder
            dimension=config.get(
                "dimension"
            ),  # Dimension of the intermediate representation
            ratios=config.get("strides"),  # Stride ratios for downsampling
            lstm=config.get("lstm_layers"),  # Number of LSTM layers (if any)
            bidirectional=config.get("bidirectional"),  # Whether LSTM is bidirectional
            # transformer_layers=config.get("transformer_layers"),  # (Optional) Number of transformer layers
            # num_heads=config.get("num_heads"),  # (Optional) Number of attention heads in transformers
            # dim_feedforward=config.get("dim_feedforward"),  # (Optional) Dimension of feedforward layer in transformers
            # dropout=config.get("dropout"),  # (Optional) Dropout rate in transformers
            dilation_base=config.get(
                "dilation_base"
            ),  # Base dilation factor for convolutions
            residual_kernel_size=config.get(
                "residual_kernel_size"
            ),  # Kernel size for residual blocks
            n_residual_layers=config.get(
                "n_residual_layers"
            ),  # Number of residual layers
            activation=config.get(
                "activation"
            ),  # Activation function for residual blocks
        )

        # Store the sample rate and quantizer parameters
        self.sample_rate = config.get("sample_rate")
        self.n_q = config.get("n_q")
        self.downsample_rate = np.prod(
            config.get("strides")
        )  # Calculate downsampling rate from stride ratios

        # Linear transformation to adjust the dimension of intermediate features if needed
        if config.get("dimension") != config.get("semantic_dimension") + config.get(
            "vocal_dimension", 0
        ):
            self.transform = nn.Linear(
                config.get("dimension"),
                config.get("semantic_dimension") + config.get("vocal_dimension", 0),
            )
        else:
            self.transform = nn.Identity()  # No transformation if dimensions match

        # Initialize the residual vector quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.get(
                "dimension"
            ),  # Dimension of the input to the quantizer
            n_q=config.get("n_q"),  # Number of quantization levels
            bins=config.get("codebook_size"),  # Size of the codebook
        )

        # Initialize the decoder part of the model with parameters from the config.
        self.decoder = SEANetDecoder(
            n_filters=config.get("n_filters"),  # Number of filters for the decoder
            dimension=config.get(
                "dimension"
            ),  # Dimension of the intermediate representation
            ratios=config.get("strides"),  # Stride ratios for upsampling
            lstm=config.get("lstm_layers"),  # Number of LSTM layers (if any)
            bidirectional=False,  # Decoder LSTM is not bidirectional
            # transformer_layers=config.get("transformer_layers"),  # (Optional) Number of transformer layers
            # num_heads=config.get("num_heads"),  # (Optional) Number of attention heads in transformers
            # dim_feedforward=config.get("dim_feedforward"),  # (Optional) Dimension of feedforward layer in transformers
            # dropout=config.get("dropout"),  # (Optional) Dropout rate in transformers
            dilation_base=config.get(
                "dilation_base"
            ),  # Base dilation factor for convolutions
            residual_kernel_size=config.get(
                "residual_kernel_size"
            ),  # Kernel size for residual blocks
            n_residual_layers=config.get(
                "n_residual_layers"
            ),  # Number of residual layers
            activation=config.get(
                "activation"
            ),  # Activation function for residual blocks
        )

    @classmethod
    def load_from_checkpoint(
        cls, config_path: str, ckpt_path: str, map_location: str = "cpu"
    ):
        """
        Load the SpeechTokenizer model from a checkpoint.

        Parameters
        ----------
        config_path : str
            Path to the model configuration file.
        ckpt_path : str
            Path to the model checkpoint file.

        Returns
        -------
        model : SpeechTokenizer
            An instance of the SpeechTokenizer model with loaded weights.
        """
        import json

        # Load model configuration from JSON file
        with open(config_path) as f:
            cfg = json.load(f)

        # Create an instance of the model using the configuration
        model = cls(cfg)

        # Load the model state from the checkpoint file
        params = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(params)

        return model

    def forward(
        self,
        x: torch.tensor,
        n_q: int = None,
        layers: list = [0],
    ):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.tensor
            Input audio waveform. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used for encoding. If None, use default.
        layers : list[int], optional
            List of layers for RVQ to return quantized results. Default is the first layer.

        Returns
        -------
        o : torch.tensor
            Reconstructed audio waveform. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from the residual vector quantizers.
        feature : torch.tensor
            Output features of the first layer of RVQ. Shape: (batch, timesteps, dimension)
        """
        # Use provided n_q or default to the model's n_q
        n_q = n_q if n_q else self.n_q

        # Encode input through the encoder
        e = self.encoder(x)

        # Quantize the encoded features
        quantized, codes, commit_loss, quantized_list, _ = self.quantizer(
            e, n_q=n_q, layers=layers
        )

        # Rearrange features and apply transformation if needed
        feature = rearrange(quantized_list[0], "b d t -> b t d")
        feature = self.transform(feature)

        # Decode the quantized features to reconstruct the input
        o = self.decoder(quantized)

        return o, commit_loss, feature

    def forward_feature(
        self,
        x: torch.tensor,
        layers: list = None,
    ):
        """
        Forward pass to get features from specific RVQ layers.

        Parameters
        ----------
        x : torch.tensor
            Input audio waveform. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            List of layers for RVQ to return quantized results. If None, use all layers.

        Returns
        -------
        quantized_list : list[torch.tensor]
            Quantized outputs from the specified layers.
        """
        # Encode input through the encoder
        e = self.encoder(x)

        # Use provided layers or default to all layers
        layers = layers if layers else list(range(self.n_q))

        # Quantize the encoded features
        _, _, _, quantized_list, _ = self.quantizer(e, layers=layers)

        return quantized_list

    def encode(
        self,
        x: torch.tensor,
        n_q: int = None,
        st: int = None,
    ):
        """
        Encode input audio into quantization codes.

        Parameters
        ----------
        x : torch.tensor
            Input audio waveform. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used for encoding. If None, use default.
        st : int, optional
            Start quantizer index for RVQ. Default is 0.

        Returns
        -------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps)
        """
        # Encode input through the encoder
        e = self.encoder(x)

        # Use provided start index or default to 0
        st = st if st is not None else 0

        # Encode the features into quantization codes
        codes = self.quantizer.encode(e, n_q=n_q, st=st)

        return codes

    def decode(self, codes: torch.tensor, st: int = 0):
        """
        Decode quantization codes into reconstructed audio.

        Parameters
        ----------
        codes : torch.tensor
            Quantization codes. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index for RVQ. Default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstructed audio waveform from codes. Shape: (batch, channels, timesteps)
        """
        # Decode the quantization codes into features
        quantized = self.quantizer.decode(codes, st=st)

        # Decode the features to reconstruct the input
        o = self.decoder(quantized)

        return o


class ExportableSpeechTokenizer(nn.Module):
    """
    Inference-only version of SpeechTokenizer using StaticResidualVectorQuantizer.
    """

    def __init__(self, pretrained_model: SpeechTokenizer):
        super().__init__()

        # Reuse encoder, decoder, transform directly
        self.encoder = pretrained_model.encoder
        self.decoder = pretrained_model.decoder
        self.transform = pretrained_model.transform

        # Extract frozen codebooks from each quantizer stage
        codebooks = [
            layer._codebook.embed.detach().clone()
            for layer in pretrained_model.quantizer.vq.layers
        ]
        self.quantizer = StaticResidualVectorQuantizer(codebooks)

    def forward(self, x: torch.Tensor):
        e = self.encoder(x)  # (B, D, T)
        q = self.quantizer(e)  # (B, D, T)
        f = self.transform(q.transpose(1, 2)).transpose(1, 2)  # (B, D', T)
        o = self.decoder(q)  # (B, C, T)
        return o, f


def load_exportable_model(
    config_path: str, ckpt_path: str, map_location="cpu"
) -> nn.Module:
    base_model = SpeechTokenizer.load_from_checkpoint(
        config_path, ckpt_path, map_location
    )
    return ExportableSpeechTokenizer(base_model).eval()

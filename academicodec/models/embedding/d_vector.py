import json

import torch
import torch.nn as nn


# CNN-LSTM Model for vocal effort classification
class CNN_LSTM(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(CNN_LSTM, self).__init__()
        self.channels = config.get("channels")
        self.kernels = config.get("kernels")
        self.strides = config.get("strides")
        self.paddings = config.get("paddings")
        self.pools = config.get("pools")
        self.hidden = config.get("hidden")
        self.bi = config.get("bidirectional")
        self.activation = config.get("activation")
        self.mfcc = config.get("n_mfcc")
        self.classe = config.get("n_classe")
        # CNN layers for local feature extraction
        self.conv1 = nn.Conv2d(
            1,
            self.channels[0],
            kernel_size=self.kernels[0],
            stride=self.strides[0],
            padding=self.paddings[0],
        )
        self.conv2 = nn.Conv2d(
            self.channels[0],
            self.channels[1],
            kernel_size=self.kernels[1],
            stride=self.strides[1],
            padding=self.paddings[1],
        )
        self.conv3 = nn.Conv2d(
            self.channels[1],
            self.channels[2],
            kernel_size=self.kernels[2],
            stride=self.strides[2],
            padding=self.paddings[2],
        )
        self.pool = nn.MaxPool2d(self.pools[0], self.pools[1])

        # LSTM layer to capture temporal dynamics
        input_size = self.mfcc * 3 // 2 // 2 // 2 * self.channels[-1]
        self.lstm = nn.LSTM(
            input_size, self.hidden, batch_first=True, bidirectional=self.bi
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(self.hidden, self.classe)

        self.activation_function = getattr(nn.functional, self.activation)

    @classmethod
    def load_from_checkpoint(
        cls, config_path: str, ckpt_path: str, map_location: str = "cpu"
    ):

        # Load model configuration from JSON file
        with open(config_path) as f:
            cfg = json.load(f)

        # Create an instance of the model using the configuration
        model = cls(cfg)

        # Load the model state from the checkpoint file
        params = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(params)

        return model

    def forward_feature(self, x):
        # CNN part
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten the output for LSTM input
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, width, -1)

        # LSTM part
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step

        return lstm_out

    def forward(self, x):
        lstm_out = self.forward_feature(x)

        # Fully connected layer
        out = self.activation_function(self.fc(lstm_out))

        return out

import json

import torch
import torch.nn as nn


# CNN-LSTM Model for vocal effort classification
class SPLPredictorCNN(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(SPLPredictorCNN, self).__init__()
        self.channels = config.get("channels")
        self.kernels = config.get("kernels")
        self.strides = config.get("strides")
        self.paddings = config.get("paddings")
        self.pools = config.get("pools")
        self.hidden = config.get("hidden")
        self.activation = config.get("activation")
        self.output = config.get("output")
        self.proba = config.get("proba")
        self.classe = config.get("n_classe")
        self.bi = config.get("bidirectional")

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
        # self.conv4 = nn.Conv2d(
        #     self.channels[2],
        #     self.channels[3],
        #     kernel_size=self.kernels[3],
        #     stride=self.strides[3],
        #     padding=self.paddings[3],
        # )
        self.pool = nn.MaxPool2d(self.pools[0], self.pools[1])

        # LSTM layer to capture temporal dynamics
        self.lstm = nn.LSTM(
            self.hidden[0], self.hidden[1], batch_first=True, bidirectional=self.bi
        )

        # self.lstm1 = nn.LSTM(
        #     self.hidden[0], self.hidden[0], batch_first=True, bidirectional=self.bi
        # )
        # self.lstm2 = nn.LSTM(
        #     self.hidden[0], self.hidden[0], batch_first=True, bidirectional=self.bi
        # )
        # self.lstm3 = nn.LSTM(
        #     self.hidden[0], self.hidden[0], batch_first=True, bidirectional=self.bi
        # )

        # Fully connected layer for classification
        self.fc1 = nn.Linear(self.hidden[1], self.hidden[2])
        self.fc2 = nn.Linear(self.hidden[2], self.classe)

        # self.fc1 = nn.Linear(self.hidden[1], self.hidden[2])
        # self.fc2 = nn.Linear(self.hidden[1], self.hidden[2])
        # self.fc3 = nn.Linear(self.hidden[2], self.classe)

        self.drop = nn.Dropout(self.proba)

        self.activation_function = getattr(nn, self.activation)()
        self.output_function = getattr(nn, self.output)()

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

    def forward_cnn(self, x):
        # CNN part
        # print(x.shape)
        x = self.pool(self.activation_function(self.conv1(x)))
        # print(x.shape)
        x = self.pool(self.activation_function(self.conv2(x)))
        # print(x.shape)
        x = self.pool(self.activation_function(self.conv3(x)))
        # x = self.pool(self.activation_function(self.conv4(x)))

        # Flatten the output for LSTM input
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, width, -1)

        return x

    def forward_lstm(self, x):
        x = self.forward_cnn(x)
        # LSTM part
        x, _ = self.lstm(x)

        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take the last time step

        return x

    def forward(self, x):
        x = self.forward_lstm(x)
        # Fully connected layer
        x = self.activation_function(self.fc1(x))
        x = self.drop(x)
        out = self.output_function(self.fc2(x))

        # x = self.activation_function(self.fc1(x))
        # x = self.drop(x)
        # x = self.activation_function(self.fc2(x))
        # x = self.drop(x)
        # out = self.output_function(self.fc3(x))

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F  # kept for compatibility if you use F.* elsewhere


class SimpleNN(nn.Module):
    def __init__(self, hidden_structure=None, use_dropout: bool = False, dropout_p: float = 0.3):
        """
        Initializes a simple MLP for MNIST.

        Args:
            hidden_structure (list[int], optional):
                A list of integers defining the size of each hidden layer.
                If None, uses the default [128, 64].
            use_dropout (bool, optional):
                If True, inserts nn.Dropout(dropout_p) after each hidden ReLU. Default: False.
            dropout_p (float, optional):
                Dropout probability used when use_dropout=True. Default: 0.3.
        """
        super(SimpleNN, self).__init__()

        # 1) Default structure
        if hidden_structure is None:
            hidden_structure = [128, 64]

        self.use_dropout = bool(use_dropout)
        self.dropout_p = float(dropout_p)

        self.flatten = nn.Flatten()

        # 2) Dynamically build the hidden block
        layers = []
        in_features = 28 * 28  # MNIST fixed input
        for out_features in hidden_structure:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if self.use_dropout and self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
            in_features = out_features

        self.hidden_layers = nn.Sequential(*layers)

        # 3) Output layer to 10 classes
        self.output_layer = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class ConvNN(nn.Module):
    def __init__(
        self,
        conv_channels=None,
        linear_hidden=None,
        input_dims=(1, 28, 28),
        num_classes=10,
        use_dropout: bool = False,
        dropout_p: float = 0.3
    ):
        """
        Initializes a flexible convolutional network for MNIST.

        Args:
            conv_channels (list[int], optional):
                Output channels per Conv block, e.g. [16, 32]. Default: [16, 32].
            linear_hidden (list[int], optional):
                Hidden sizes for the linear classifier block, e.g. [128]. Default: [128].
            input_dims (tuple, optional):
                (C, H, W) input shape. Default: (1, 28, 28) for MNIST.
            num_classes (int, optional):
                Number of output classes. Default: 10.
            use_dropout (bool, optional):
                If True, applies nn.Dropout2d(p) after each conv ReLU, and nn.Dropout(p)
                after each linear ReLU. Default: False.
            dropout_p (float, optional):
                Dropout probability used when use_dropout=True. Default: 0.3.
        """
        super(ConvNN, self).__init__()

        # --- 1) Defaults ---
        if conv_channels is None:
            conv_channels = [16, 32]
        if linear_hidden is None:
            linear_hidden = [128]

        self.num_classes = num_classes
        self.use_dropout = bool(use_dropout)
        self.dropout_p = float(dropout_p)

        in_channels, h, w = input_dims

        # --- 2) Convolutional feature extractor ---
        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2))
            conv_layers.append(nn.ReLU())
            if self.use_dropout and self.dropout_p > 0:
                # Spatial dropout drops entire feature maps (channels) at once
                conv_layers.append(nn.Dropout2d(p=self.dropout_p))
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        # --- 3) Compute linear input size via a dummy forward ---
        with torch.no_grad():
            dummy = torch.randn(1, *input_dims)
            conv_out = self.conv_block(dummy)
            linear_in_features = conv_out.numel()

        # --- 4) Linear classifier block ---
        linear_layers = []
        in_features = linear_in_features
        for out_features in linear_hidden:
            linear_layers.append(nn.Linear(in_features, out_features))
            linear_layers.append(nn.ReLU())
            if self.use_dropout and self.dropout_p > 0:
                linear_layers.append(nn.Dropout(p=self.dropout_p))
            in_features = out_features

        self.classifier_block = nn.Sequential(*linear_layers)

        # --- 5) Final output layer ---
        self.output_layer = nn.Linear(in_features, self.num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_block(x)
        x = self.output_layer(x)
        return x

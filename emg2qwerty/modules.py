# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class LSTMEncoder(nn.Module):
    """An LSTM-based encoder that processes the input sequence.
    
    Args:
        num_features (int): Number of input features
        hidden_size (int): Hidden size of the LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability between LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        proj_size (int | None): Optional projection size to reduce LSTM output dimension
    """

    def __init__(
        self,
        num_features: int,
        lstm_hidden_size: int = 512,
        num_layers: int = 4,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.lstm_layers = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=True,
        )

        # Output projection if needed
        self.fc_block = TDSFullyConnectedBlock(lstm_hidden_size*2)
        self.out_layer = nn.Linear(lstm_hidden_size*2, num_features)

    def _forward_impl(self, x, h=None):
        # Separate function for checkpointing
        x, _ = self.lstm_layers(x, h)
        x = self.fc_block(x)
        x = self.out_layer(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Use checkpointing during training
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, inputs, None)
        else:
            # Regular forward pass during inference
            return self._forward_impl(inputs)


class GRUEncoder(nn.Module):
    """A GRU-based encoder that processes the input sequence.
    
    Args:
        num_features (int): Number of input features
        gru_hidden_size (int): Hidden size of the GRU
        num_layers (int): Number of GRU layers
        dropout (float): Dropout probability between GRU layers
    """

    def __init__(
        self,
        num_features: int,
        gru_hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # FC block for processing GRU output
        self.fc_block = TDSFullyConnectedBlock(gru_hidden_size * 2)
        
        # Final projection back to input dimension
        self.out_layer = nn.Linear(gru_hidden_size * 2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        
        # Apply GRU
        x, _ = self.gru(inputs)  # (T, N, hidden_size * 2)
        
        # Apply FC block with residual connection and layer norm
        x = self.fc_block(x)
        
        # Project back to input dimension
        x = self.out_layer(x)
        
        return x
    
#used with tranformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=150000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    """A Transformer-based encoder that processes the input sequence.
    
    Args:
        num_features (int): Number of input features
        d_model (int): Transformer model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Project input to transformer dimension
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=150000)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        #self.fc_block = TDSFullyConnectedBlock(d_model)
        #self.out_layer = nn.Linear(d_model, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        
        # Project to transformer dimension
        x = self.input_proj(inputs)  # (T, N, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer(x)  # (T, N, d_model)
        
        # Process and project back to input dimension
        # x = self.fc_block(x)
        # x = self.out_layer(x)  # (T, N, num_features)
        
        return x



class CNNRNNEncoder(nn.Module):
    """A hybrid encoder that combines CNN for feature extraction with RNN for sequence processing.
    
    Args:
        num_features (int): Number of input features
        cnn_channels (list): List of CNN channel sizes
        kernel_size (int): CNN kernel size
        rnn_hidden_size (int): Hidden size of the RNN
        rnn_num_layers (int): Number of RNN layers
        rnn_type (str): Type of RNN ('lstm' or 'gru')
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        num_features: int,
        cnn_channels: list[int] = [64, 128, 256],
        kernel_size: int = 3,
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 4,
        rnn_type: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = num_features
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels, 
                    out_channels,
                    kernel_size,
                    padding=kernel_size//2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # RNN for sequence processing
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=dropout if rnn_num_layers > 1 else 0,
            bidirectional=True,
            batch_first=False
        )
        
        # Output processing
        self.fc_block = TDSFullyConnectedBlock(rnn_hidden_size * 2)
        self.out_layer = nn.Linear(rnn_hidden_size * 2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        
        # Prepare for CNN
        T, N, C = inputs.shape
        x = inputs.permute(1, 2, 0)  # (N, C, T)
        
        # Apply CNN
        x = self.cnn(x)  # (N, cnn_channels[-1], T)
        
        # Prepare for RNN
        x = x.permute(2, 0, 1)  # (T, N, cnn_channels[-1])
        
        # Apply RNN
        x, _ = self.rnn(x)  # (T, N, hidden_size * 2)
        
        # Process output
        x = self.fc_block(x)
        x = self.out_layer(x)  # (T, N, num_features)
        
        return x

class TransformerWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_features=embedding_dim,  # Use embedding dimension
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):
        # Ensure inputs are of type LongTensor
        if inputs.dtype != torch.long:
            inputs = inputs.long()
        
        # Convert inputs to embeddings
        x = self.embedding(inputs)  # (T, N, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Project to vocab size
        x = self.output_layer(x)  # (T, N, vocab_size)
        
        return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Create a long enough 'pe' matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

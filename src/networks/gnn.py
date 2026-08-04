"""Graph Neural Network (GNN) models for multivariate time series."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions, build_gnn_node_groups, DEFAULT_VARIABLE_NODE_GROUPS
from ._registry import register_model
from .transformer import PositionalEncoding
from .tcn import TemporalBlock
from .patchtst import PatchEmbedding
from .timesnet import TimesBlock


class GraphConvLayer(nn.Module):
    """Single graph convolution layer.

    Performs message passing on a graph with adaptive adjacency matrix.
    X' = sigma(A @ X @ W + b)

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (batch, num_nodes, in_features).
            adj: Adjacency matrix (num_nodes, num_nodes), softmax-normalized.

        Returns:
            Updated node features (batch, num_nodes, out_features).
        """
        # Message passing: aggregate neighbor features via adjacency
        support = torch.matmul(adj, x)  # (batch, num_nodes, in_features)
        return self.weight(support)  # (batch, num_nodes, out_features)


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for multivariate time series.

    Maps input variables to graph nodes, learns adaptive inter-variable
    relationships via GCN, then applies a temporal encoder (Transformer,
    TCN, or BiLSTM) to capture time dynamics.

    Architecture:
        1. Variable grouping: 23 input vars -> 9 graph nodes
        2. Per-timestep GCN: learns inter-variable relationships
        3. Temporal encoder: captures time dynamics
        4. Output projection: -> d_model features

    Args:
        num_input_variables: Number of input variables (23).
        input_sequence_length: Length of input sequence.
        num_nodes: Number of graph nodes (default 9).
        node_feature_dim: Feature dimension per node after projection.
        gcn_hidden_dim: Hidden dimension in GCN layers.
        num_gcn_layers: Number of GCN layers.
        temporal_type: Temporal encoder type ("transformer", "tcn", "bilstm",
            "lstm", "patch_transformer", "timesnet", "linear").
        d_model: Output feature dimension.
        dropout: Dropout rate.
        node_embed_dim: Dimension of node embeddings for adaptive adjacency.
        transformer_nhead: Number of attention heads (for transformer temporal).
        transformer_num_layers: Number of transformer layers.
        transformer_dim_feedforward: Feedforward dimension.
        tcn_channels: Channel list for TCN temporal encoder.
        tcn_kernel_size: Kernel size for TCN.
        bilstm_hidden_size: Hidden size for BiLSTM / LSTM.
        bilstm_num_layers: Number of BiLSTM / LSTM layers.
        timesnet_d_ff: Hidden dim in TimesNet Inception blocks.
        timesnet_num_blocks: Number of stacked TimesBlocks.
        timesnet_top_k: Number of dominant periods per TimesBlock.
        timesnet_num_kernels: Number of Inception conv branches.
        linear_hidden_dim: Hidden dim for the Flatten+MLP temporal reducer.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        group_sizes: list = None,
        num_nodes: int = None,
        node_feature_dim: int = 32,
        gcn_hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        temporal_type: str = "transformer",
        d_model: int = 128,
        dropout: float = 0.1,
        node_embed_dim: int = 16,
        # Transformer temporal params
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        # TCN temporal params
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        # BiLSTM / LSTM temporal params
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        # PatchTransformer temporal params
        patch_len: int = 16,
        patch_stride: int = 8,
        # TimesNet temporal params
        timesnet_d_ff: int = 128,
        timesnet_num_blocks: int = 2,
        timesnet_top_k: int = 3,
        timesnet_num_kernels: int = 3,
        # Linear (Flatten+MLP) temporal params
        linear_hidden_dim: int = 256,
    ):
        super().__init__()

        # Use provided group_sizes or fallback defaults
        if group_sizes is None:
            group_sizes = [len(v) for v in DEFAULT_VARIABLE_NODE_GROUPS.values()]
        if num_nodes is None:
            num_nodes = len(group_sizes)

        if num_input_variables != sum(group_sizes):
            raise ValueError(
                f"num_input_variables ({num_input_variables}) != "
                f"sum(group_sizes) ({sum(group_sizes)})"
            )

        self._GROUP_SIZES = group_sizes
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.temporal_type = temporal_type
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length

        # Per-node input projections (variable group -> node_feature_dim)
        self.node_projections = nn.ModuleList()
        for size in self._GROUP_SIZES:
            self.node_projections.append(
                nn.Linear(size, node_feature_dim)
            )

        # Adaptive adjacency matrix via learnable node embeddings
        # A = softmax(relu(E1 @ E2^T))
        self.node_embed1 = nn.Parameter(
            torch.randn(num_nodes, node_embed_dim)
        )
        self.node_embed2 = nn.Parameter(
            torch.randn(num_nodes, node_embed_dim)
        )

        # GCN layers
        gcn_layers = []
        in_dim = node_feature_dim
        for i in range(num_gcn_layers):
            out_dim = gcn_hidden_dim if i < num_gcn_layers - 1 else gcn_hidden_dim
            gcn_layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_activation = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout)

        # Flatten GCN output: num_nodes * gcn_hidden_dim -> temporal input dim
        temporal_input_dim = num_nodes * gcn_hidden_dim

        # Temporal encoder (processes GCN features across time)
        if temporal_type == "transformer":
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self.pos_encoder = PositionalEncoding(
                d_model, input_sequence_length, dropout
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )
        elif temporal_type == "tcn":
            if tcn_channels is None:
                tcn_channels = [64, 128, 256]
            self.temporal_proj = nn.Linear(temporal_input_dim, tcn_channels[0])
            layers = []
            num_ch = [tcn_channels[0]] + list(tcn_channels)
            for i in range(len(tcn_channels)):
                layers.append(TemporalBlock(
                    num_ch[i], num_ch[i + 1], tcn_kernel_size,
                    dilation=2 ** i, dropout=dropout
                ))
            self.temporal_encoder = nn.Sequential(*layers)
            self._tcn_out_dim = tcn_channels[-1]
        elif temporal_type == "bilstm":
            self.temporal_proj = nn.Linear(temporal_input_dim, bilstm_hidden_size)
            self.temporal_encoder = nn.LSTM(
                input_size=bilstm_hidden_size,
                hidden_size=bilstm_hidden_size,
                num_layers=bilstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if bilstm_num_layers > 1 else 0.0
            )
            self._bilstm_out_dim = bilstm_hidden_size * 2  # bidirectional
        elif temporal_type == "lstm":
            self.temporal_proj = nn.Linear(temporal_input_dim, bilstm_hidden_size)
            self.temporal_encoder = nn.LSTM(
                input_size=bilstm_hidden_size,
                hidden_size=bilstm_hidden_size,
                num_layers=bilstm_num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if bilstm_num_layers > 1 else 0.0
            )
            self._lstm_out_dim = bilstm_hidden_size  # unidirectional
        elif temporal_type == "timesnet":
            # FFT period detection requires top_k <= seq_len // 2
            effective_top_k = min(timesnet_top_k, max(input_sequence_length // 2, 1))
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self._timesnet_blocks = nn.ModuleList([
                TimesBlock(
                    seq_len=input_sequence_length,
                    d_model=d_model,
                    d_ff=timesnet_d_ff,
                    top_k=effective_top_k,
                    num_kernels=timesnet_num_kernels,
                )
                for _ in range(timesnet_num_blocks)
            ])
            self._timesnet_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(timesnet_num_blocks)
            ])
            self._timesnet_dropouts = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(timesnet_num_blocks)
            ])
            self.temporal_encoder = None  # branch uses dedicated block list
        elif temporal_type == "linear":
            # Flatten+MLP temporal reducer: drops sequence structure
            flat_dim = input_sequence_length * temporal_input_dim
            self.temporal_proj = None  # unused in this branch
            self.temporal_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, linear_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(linear_hidden_dim, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif temporal_type == "patch_transformer":
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self._patch_embed = PatchEmbedding(
                patch_len=patch_len,
                stride=patch_stride,
                d_input=d_model,
                d_model=d_model,
                dropout=dropout
            )
            # Calculate num_patches for positional embedding
            pad_len = (patch_stride - (input_sequence_length - patch_len) % patch_stride) % patch_stride
            n_patches = (input_sequence_length + pad_len - patch_len) // patch_stride + 1
            self._patch_pos_embed = nn.Parameter(
                torch.randn(1, n_patches, d_model) * 0.02
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        # Global pooling + output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if temporal_type in ("transformer", "patch_transformer", "timesnet"):
            self.output_projection = nn.Linear(d_model, d_model)
        elif temporal_type == "tcn":
            self.output_projection = nn.Linear(self._tcn_out_dim, d_model)
        elif temporal_type == "bilstm":
            self.output_projection = nn.Linear(self._bilstm_out_dim, d_model)
        elif temporal_type == "lstm":
            self.output_projection = nn.Linear(self._lstm_out_dim, d_model)
        elif temporal_type == "linear":
            # Already produces d_model from the MLP; identity keeps API parity
            self.output_projection = nn.Identity()

    def _compute_adaptive_adj(self) -> torch.Tensor:
        """Compute adaptive adjacency matrix from node embeddings."""
        adj = F.relu(torch.matmul(self.node_embed1, self.node_embed2.T))
        return F.softmax(adj, dim=1)

    def _split_to_nodes(self, x: torch.Tensor) -> list:
        """Split input variables into node groups.

        Args:
            x: Input tensor (batch, seq_len, 23).

        Returns:
            List of 9 tensors, each (batch, seq_len, group_size).
        """
        nodes = []
        idx = 0
        for size in self._GROUP_SIZES:
            nodes.append(x[:, :, idx:idx + size])
            idx += size
        return nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_vars).

        Returns:
            Output features (batch, d_model).
        """
        batch_size, seq_len, _ = x.size()

        # 1. Split variables into node groups and project
        node_groups = self._split_to_nodes(x)
        node_features = []
        for i, group in enumerate(node_groups):
            # (batch, seq_len, group_size) -> (batch, seq_len, node_feature_dim)
            node_features.append(self.node_projections[i](group))
        # Stack: (batch, seq_len, num_nodes, node_feature_dim)
        node_features = torch.stack(node_features, dim=2)

        # 2. Compute adaptive adjacency matrix
        adj = self._compute_adaptive_adj()  # (num_nodes, num_nodes)

        # 3. Apply GCN per timestep
        # Reshape: (batch * seq_len, num_nodes, node_feature_dim)
        h = node_features.reshape(batch_size * seq_len, self.num_nodes, -1)
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj)
            h = self.gcn_activation(h)
            h = self.gcn_dropout(h)
        # (batch * seq_len, num_nodes, gcn_hidden_dim)

        # 4. Flatten nodes: -> (batch, seq_len, num_nodes * gcn_hidden_dim)
        h = h.reshape(batch_size, seq_len, -1)

        # 5. Temporal encoding
        if self.temporal_type == "transformer":
            h = self.temporal_proj(h)  # (batch, seq_len, d_model)
            h = self.pos_encoder(h)
            h = self.temporal_encoder(h)  # (batch, seq_len, d_model)
            h = h.transpose(1, 2)  # (batch, d_model, seq_len)
        elif self.temporal_type == "tcn":
            h = self.temporal_proj(h)  # (batch, seq_len, tcn_channels[0])
            h = h.transpose(1, 2)  # (batch, tcn_channels[0], seq_len)
            h = self.temporal_encoder(h)  # (batch, tcn_channels[-1], seq_len)
        elif self.temporal_type == "bilstm":
            h = self.temporal_proj(h)  # (batch, seq_len, hidden_size)
            h, _ = self.temporal_encoder(h)  # (batch, seq_len, hidden*2)
            h = h.transpose(1, 2)  # (batch, hidden*2, seq_len)
        elif self.temporal_type == "lstm":
            h = self.temporal_proj(h)  # (batch, seq_len, hidden_size)
            h, _ = self.temporal_encoder(h)  # (batch, seq_len, hidden_size)
            h = h.transpose(1, 2)  # (batch, hidden_size, seq_len)
        elif self.temporal_type == "timesnet":
            h = self.temporal_proj(h)  # (batch, seq_len, d_model)
            for block, norm, drop in zip(
                self._timesnet_blocks, self._timesnet_norms, self._timesnet_dropouts
            ):
                h = norm(drop(block(h)) + h)
            h = h.transpose(1, 2)  # (batch, d_model, seq_len)
        elif self.temporal_type == "linear":
            # Skip temporal_proj/global_pool; MLP directly reduces to d_model
            h = self.temporal_encoder(h)  # (batch, d_model)
            return self.output_projection(h)
        elif self.temporal_type == "patch_transformer":
            h = self.temporal_proj(h)  # (batch, seq_len, d_model)
            tokens = self._patch_embed(h)  # (batch, num_patches, d_model)
            tokens = tokens + self._patch_pos_embed[:, :tokens.size(1), :]
            h = self.temporal_encoder(tokens)  # (batch, num_patches, d_model)
            h = h.transpose(1, 2)  # (batch, d_model, num_patches)

        # 6. Global pooling + output projection
        h = self.global_pool(h).squeeze(-1)  # (batch, feat_dim)
        h = self.output_projection(h)  # (batch, d_model)

        return h

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """Return the learned adjacency matrix (for visualization)."""
        with torch.no_grad():
            return self._compute_adaptive_adj()


class GNNOnlyModel(nn.Module):
    """Time series model using GNN encoder with pluggable temporal backend.

    Combines graph-based inter-variable relationship learning with
    temporal sequence modeling. The temporal encoder can be swapped
    between Transformer, TCN, and BiLSTM.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence.
        num_target_variables: Number of target variables.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension.
        gnn_node_feature_dim: Feature dim per graph node.
        gnn_gcn_hidden_dim: Hidden dim in GCN layers.
        gnn_num_gcn_layers: Number of GCN layers.
        gnn_temporal_type: Temporal encoder ("transformer", "tcn", "bilstm",
            "lstm", "patch_transformer", "timesnet", "linear").
        gnn_dropout: Dropout rate.
        gnn_node_embed_dim: Dim of node embeddings for adaptive adjacency.
        transformer_nhead: Attention heads (for transformer temporal).
        transformer_num_layers: Transformer layers.
        transformer_dim_feedforward: Feedforward dim.
        tcn_channels: Channel list for TCN temporal.
        tcn_kernel_size: Kernel size for TCN.
        bilstm_hidden_size: Hidden size for BiLSTM / LSTM.
        bilstm_num_layers: Number of BiLSTM / LSTM layers.
        timesnet_d_ff: Hidden dim in TimesBlock Inception blocks.
        timesnet_num_blocks: Number of stacked TimesBlocks.
        timesnet_top_k: Number of dominant periods per TimesBlock.
        timesnet_num_kernels: Number of Inception conv branches.
        linear_hidden_dim: Hidden dim for the Flatten+MLP temporal reducer.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        gnn_group_sizes: list = None,
        gnn_num_nodes: int = None,
        gnn_node_feature_dim: int = 32,
        gnn_gcn_hidden_dim: int = 64,
        gnn_num_gcn_layers: int = 2,
        gnn_temporal_type: str = "transformer",
        gnn_dropout: float = 0.1,
        gnn_node_embed_dim: int = 16,
        # Temporal encoder params (passed through)
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        patch_len: int = 16,
        patch_stride: int = 8,
        timesnet_d_ff: int = 128,
        timesnet_num_blocks: int = 2,
        timesnet_top_k: int = 3,
        timesnet_num_kernels: int = 3,
        linear_hidden_dim: int = 256,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.gnn_encoder = GNNEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            group_sizes=gnn_group_sizes,
            num_nodes=gnn_num_nodes,
            node_feature_dim=gnn_node_feature_dim,
            gcn_hidden_dim=gnn_gcn_hidden_dim,
            num_gcn_layers=gnn_num_gcn_layers,
            temporal_type=gnn_temporal_type,
            d_model=d_model,
            dropout=gnn_dropout,
            node_embed_dim=gnn_node_embed_dim,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            bilstm_hidden_size=bilstm_hidden_size,
            bilstm_num_layers=bilstm_num_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            timesnet_d_ff=timesnet_d_ff,
            timesnet_num_blocks=timesnet_num_blocks,
            timesnet_top_k=timesnet_top_k,
            timesnet_num_kernels=timesnet_num_kernels,
            linear_hidden_dim=linear_hidden_dim,
        )

        # Regression head (identical to Transformer/TCN models)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(gnn_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """Return the learned adjacency matrix (for visualization)."""
        return self.gnn_encoder.adjacency_matrix

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass.

        Args:
            solar_wind_input: Input time series (B, seq_len, num_vars).
            image_input: Ignored (API compatibility).
            return_features: Whether to return intermediate features.

        Returns:
            Predictions (B, target_seq_len, num_target_vars), or
            tuple (predictions, features, None) if return_features=True.
        """
        gnn_features = self.gnn_encoder(solar_wind_input)  # (B, d_model)

        predictions = self.regression_head(gnn_features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables
        )

        if return_features:
            return output, gnn_features, None
        return output


@register_model("gnn")
def _create_gnn(config):
    """Factory function for GNN model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating gnn model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    # GNN model with pluggable temporal encoder
    # Build node groups dynamically from config (with validation)
    gnn_group_sizes, gnn_num_nodes = build_gnn_node_groups(config)

    gnn_temporal_type = getattr(config.model, 'gnn_temporal_type', 'transformer')
    gnn_node_feature_dim = getattr(config.model, 'gnn_node_feature_dim', 32)
    gnn_gcn_hidden_dim = getattr(config.model, 'gnn_gcn_hidden_dim', 64)
    gnn_num_gcn_layers = getattr(config.model, 'gnn_num_gcn_layers', 2)
    gnn_dropout = getattr(config.model, 'gnn_dropout', 0.1)
    gnn_node_embed_dim = getattr(config.model, 'gnn_node_embed_dim', 16)

    # Temporal encoder params (reuse existing config keys)
    tcn_channels = getattr(config.model, 'tcn_channels', [64, 128, 256])
    if hasattr(tcn_channels, '__iter__') and not isinstance(tcn_channels, list):
        tcn_channels = list(tcn_channels)
    tcn_kernel_size = getattr(config.model, 'tcn_kernel_size', 3)
    bilstm_hidden_size = getattr(config.model, 'bilstm_hidden_size', 128)
    bilstm_num_layers = getattr(config.model, 'bilstm_num_layers', 2)
    patch_len = getattr(config.model, 'patch_len', 16)
    patch_stride = getattr(config.model, 'patch_stride', 8)
    timesnet_d_ff = getattr(config.model, 'timesnet_d_ff', 128)
    timesnet_num_blocks = getattr(config.model, 'timesnet_num_blocks', 2)
    timesnet_top_k = getattr(config.model, 'timesnet_top_k', 3)
    timesnet_num_kernels = getattr(config.model, 'timesnet_num_kernels', 3)
    linear_hidden_dim = getattr(config.model, 'gnn_linear_hidden_dim', 256)

    model = GNNOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        gnn_group_sizes=gnn_group_sizes,
        gnn_num_nodes=gnn_num_nodes,
        gnn_node_feature_dim=gnn_node_feature_dim,
        gnn_gcn_hidden_dim=gnn_gcn_hidden_dim,
        gnn_num_gcn_layers=gnn_num_gcn_layers,
        gnn_temporal_type=gnn_temporal_type,
        gnn_dropout=gnn_dropout,
        gnn_node_embed_dim=gnn_node_embed_dim,
        transformer_nhead=config.model.transformer_nhead,
        transformer_num_layers=config.model.transformer_num_layers,
        transformer_dim_feedforward=config.model.transformer_dim_feedforward,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        bilstm_hidden_size=bilstm_hidden_size,
        bilstm_num_layers=bilstm_num_layers,
        patch_len=patch_len,
        patch_stride=patch_stride,
        timesnet_d_ff=timesnet_d_ff,
        timesnet_num_blocks=timesnet_num_blocks,
        timesnet_top_k=timesnet_top_k,
        timesnet_num_kernels=timesnet_num_kernels,
        linear_hidden_dim=linear_hidden_dim,
    )
    print(f"  GNN temporal encoder: {gnn_temporal_type}")
    print(f"  GNN: {gnn_num_gcn_layers} GCN layers, {gnn_num_nodes} nodes, "
          f"groups={gnn_group_sizes}")
    return model

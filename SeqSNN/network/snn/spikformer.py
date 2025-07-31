from typing import Optional

from pathlib import Path
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from ..base import NETWORKS
from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ...module.spike_attention import Block

from ...module.clustering import Cluster_assigner

tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, D, L
        enc = self.encoder(inputs)  # B, T, D, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, D, L
        spks = self.lif(enc)  # T, B, D, L
        return spks


@NETWORKS.register_module("Spikformer")
class Spikformer(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 5000,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
        use_cluster: bool = False,
        use_ste: bool = False,  # Use Straight-Through Estimator for cluster probabilities
        use_all_zero: bool = False,  # Use all-zero cluster probabilities
        use_all_random: bool = False,  # Use all-random cluster probabilities
        gpu_id: Optional[int] = None,
        n_cluster: Optional[int] = 3,  # Number of clusters for clustering
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.gpu_id = gpu_id
        self.use_cluster = use_cluster
        self.use_ste = use_ste
        self.n_cluster = n_cluster
        self.use_all_zero = use_all_zero
        self.use_all_random = use_all_random

        self.temporal_encoder = SpikeEncoder[self._snn_backend][encoder_type](num_steps)
        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=num_steps,
        )
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.encoder = nn.Linear(input_size + num_pe_neuron, dim)
        else:
            self.encoder = nn.Linear(input_size, dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        '''
        Cluster assigner
        '''
        if self.use_cluster:
            self.input_size = input_size
            self.max_length = max_length
            self.cluster_assigner = Cluster_assigner(
                n_vars=input_size,
                n_cluster=self.n_cluster,  # This is a dummy value, will be set later
                seq_len=max_length,
                d_model=512,
                device=self.gpu_id
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, if_update: bool = False):
        functional.reset_net(self)

        '''
        Get cluster probabilities and embeddings
        '''
        if self.use_cluster:
            cluster_prob, cluster_emb = self.cluster_assigner(
                x, self.cluster_assigner.cluster_emb
            )
            if if_update:
                self.cluster_assigner.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)

        x = self.temporal_encoder(x)  # B L C -> T B C L

        '''
        Inject cluster probabilities
        '''
        if self.use_cluster: # v1
            self.cluster_prob = cluster_prob  # [B, C, K]
            cluster_prob = cluster_prob.permute(2, 0, 1) # [K, B, C] < [B, C, K]
            cluster_prob = cluster_prob.unsqueeze(-1)  # [K, B, C, 1]
            cluster_prob = cluster_prob.repeat(1, 1, 1, x.size(3))
            cluster_prob_soft = cluster_prob
            cluster_prob_hard = torch.bernoulli(cluster_prob_soft)  # [K, B, C, L] - Bernoulli sampling
            if self.use_ste:
                cluster_prob = cluster_prob_soft + (cluster_prob_hard - cluster_prob_soft).detach()  # [K, B, C, L]
            else:
                cluster_prob = cluster_prob_soft

            x = torch.cat((x, cluster_prob), dim=0)  # T+K, B, C, L

        x = x.transpose(-2, -1)  # T B L C
        if self.pe_type != "none":
            x = self.pe(x)  # T B L C'
        T, B, L, _ = x.shape

        x = self.encoder(x.flatten(0, 1)).reshape(T, B, L, -1)  # T B L D
        x = self.init_lif(x)

        for blk in self.blocks:
            x = blk(x)  # T B L D
        out = x.mean(0)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.dim

    @property
    def hidden_size(self):
        return self.dim

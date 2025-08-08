from typing import Optional
from pathlib import Path
import torch
from torch import nn

from spikingjelly.activation_based import surrogate, neuron, functional

from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ..base import NETWORKS

from ...module.clustering import Cluster_assigner


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class SpikeRNNCell(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, output_size)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        # T, B, L, C'
        T, B, L, _ = x.shape
        x = x.flatten(0, 1)  # TB, L, C'
        x = self.linear(x)
        x = x.reshape(T, B, L, -1)
        x = self.lif(x)  # T, B, L, C'
        return x


@NETWORKS.register_module("SpikeRNN")
class SpikeRNN(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 4,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
        use_cluster: bool = False,
        use_ste: bool = False,  # Use Straight-Through Estimator for cluster probabilities
        gpu_id: Optional[int] = None,
        n_cluster: Optional[int] = 3,  # Number of clusters for clustering
        use_all_zero: bool = False,  # Use all-zero cluster probabilities
        use_all_random: bool = False,  # Use all-random cluster probabilities
        d_model: Optional[int] = 512,  # Dimension of the model for clustering
    ):
        super().__init__()
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.neuron_pe_scale = neuron_pe_scale
        self.temporal_encoder = SpikeEncoder[self._snn_backend][encoder_type](num_steps)
        self.use_cluster = use_cluster
        self.use_ste = use_ste
        self.gpu_id = gpu_id
        self.n_cluster = n_cluster
        self.use_all_zero = use_all_zero
        self.use_all_random = use_all_random

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
                d_model=d_model,
                device=self.gpu_id
            )

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.dim = hidden_size + num_pe_neuron
        else:
            self.dim = hidden_size

        if self.pe_type == "neuron" and self.pe_mode == "concat":
            self.encoder = nn.Linear(input_size + num_pe_neuron, self.dim)
        else:
            self.encoder = nn.Linear(input_size, self.dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=1.0,
            backend=backend,
        )

        self.net = nn.Sequential(
            *[
                SpikeRNNCell(input_size=self.dim, output_size=self.dim)
                for i in range(layers)
            ]
        )

        self.__output_size = self.dim

    def forward(
        self,
        inputs: torch.Tensor,
        if_update: bool = False,
    ):
        functional.reset_net(self)

        '''
        Get cluster probabilities and embeddings
        '''
        if self.use_cluster:
            cluster_prob, cluster_emb = self.cluster_assigner(
                inputs, self.cluster_assigner.cluster_emb
            )
            if if_update:
                self.cluster_assigner.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)

        hiddens = self.temporal_encoder(inputs)  # T, B, C, L

        '''
        Inject cluster probabilities
        '''
        if self.use_cluster: # v1
            self.cluster_prob = cluster_prob  # [B, C, K]
            cluster_prob = cluster_prob.permute(2, 0, 1) # [K, B, C] < [B, C, K]
            cluster_prob = cluster_prob.unsqueeze(-1)  # [K, B, C, 1]
            cluster_prob = cluster_prob.repeat(1, 1, 1, hiddens.size(3))
            cluster_prob_soft = cluster_prob
            cluster_prob_hard = torch.bernoulli(cluster_prob_soft)  # [K, B, C, L] - Bernoulli sampling
            if self.use_ste:
                cluster_prob = cluster_prob_soft + (cluster_prob_hard - cluster_prob_soft).detach()  # [K, B, C, L]
            else:
                cluster_prob = cluster_prob_soft

            if self.use_all_zero:
                cluster_prob = torch.zeros_like(cluster_prob)
                #print('check cluster_prob min-max', cluster_prob.min(), cluster_prob.max())
            elif self.use_all_random:
                assert self.use_all_zero is False, "Cannot use both all-zero and all-random cluster probabilities."
                cluster_prob = torch.rand_like(cluster_prob)
                #print('check cluster_prob min-max', cluster_prob.min(), cluster_prob.max())

            hiddens = torch.cat((hiddens, cluster_prob), dim=0)  # T+K, B, C, L

        hiddens = hiddens.transpose(-2, -1)  # T, B, L, C
        T, B, L, _ = hiddens.size()  # T, B, L, D
        if self.pe_type != "none":
            hiddens = self.pe(hiddens)  # T B L C'

        hiddens = self.encoder(hiddens.flatten(0, 1)).reshape(T, B, L, -1)  # T B L D
        hiddens = self.init_lif(hiddens)
        hiddens = self.net(hiddens)  # T, B, L, D

        out = hiddens.mean(0)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.dim


@NETWORKS.register_module("SpikeRNN2d")
class SpikeRNN2D(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        hidden_size: int,
        layers: int = 1,
        num_steps: int = 50,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
    ):
        super().__init__()
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)

        self.net = nn.Sequential(
            *[
                SpikeRNNCell(
                    hidden_size,
                    hidden_size,
                )
                for i in range(layers)
            ]
        )

        self.__output_size = hidden_size * input_size

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        for i in range(length):
            spks, mems = self.net(h[:, i, :])
        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        return spks.transpose(1, 2), spks[:, :, -1]  # B * Time Step * CH, B * CH

    @property
    def output_size(self):
        return self.__output_size

'''
TODO:
- 테스트 후 encoder.py에 통합 예정 (2025.09.23)
'''

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

from ...clustering import Cluster_assigner


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True

use_channel_shuffle = False

class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # T B L C
        inputs = inputs.permute(0, 1, 3, 2)  # T B C L
        spks = self.lif(inputs)  # T B C L

        if use_channel_shuffle:
            spks = channel_shuffle(spks, shuffle_dim=0)  # Shuffle channels
        return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # B, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # B, C, L, 1
        enc = self.enc(delta)  # B, C, L, T
        enc = enc.permute(3, 0, 1, 2)  # T, B, C, L
        spks = self.lif(enc)
        
        if use_channel_shuffle:
            spks = channel_shuffle(spks, shuffle_dim=0)  # Shuffle channels
        return spks


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
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, C, L
        enc = self.encoder(inputs)  # B, T, C, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, C, L
        spks = self.lif(enc)  # T, B, C, L

        if use_channel_shuffle:
            spks = channel_shuffle(spks, shuffle_dim=0)  # Shuffle channels
        return spks

class Cluster_wise_ConvEncoder(nn.Module):
    def __init__(self, output_size: int, channel_wise_kernel: int = 3, temporal_kernel: int = 3,
                 n_vars=321, n_cluster=3, seq_len=168, d_model=256, device='cuda'):
        super().__init__()
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

        '''
        Cluster Assigner 인스턴스 생성
        '''
        self.cluster_assigner = Cluster_assigner(
            n_vars=n_vars,
            n_cluster=n_cluster,
            seq_len=seq_len,
            d_model=d_model,
            device=device
        )

        '''
        Cluster-wise Convolution layers
        '''
        self.convs = nn.ModuleList()
        for _ in range(n_cluster):
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=(channel_wise_kernel, temporal_kernel),
                    stride=1,
                    padding=(channel_wise_kernel // 2, temporal_kernel // 2),
                ),
                nn.BatchNorm2d(output_size),
            )
            self.convs.append(conv)

        print("Cluster-wise ConvEncoder initialized with the following parameters:")
        print(f"Output Size: {output_size}")
        print(f"Channel-wise Kernel Size: {channel_wise_kernel}")
        print(f"Temporal Kernel Size: {temporal_kernel}")
        print(f"Number of Variables: {n_vars}")
        print(f"Number of Clusters: {n_cluster}")
        print(f"Sequence Length: {seq_len}")
        print(f"Model Dimension: {d_model}")

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, C

        # Cluster assignment
        cluster_prob, cluster_emb = self.cluster_assigner(inputs, self.cluster_assigner.cluster_emb, return_type='average') # [C, K], [K, d_model]

        # Channel re-ordering
        emb_distance = torch.cdist(cluster_emb, cluster_emb, p=2)  # [K, K]
        reordered_cluster_indices = self.reorder_clusters_by_distance(emb_distance)  # [K]
        cluster_prob = cluster_prob[:, reordered_cluster_indices]  # [C, K] 재정렬된 클러스터 확률

        # 각 채널의 클러스터 ID 할당
        cluster_indices = torch.argmax(cluster_prob, dim=-1)  # [C]

        # 각 클러스터 ID에 우선순위 값 부여 (reordered_cluster_indices의 역매핑)
        cluster_priority = torch.zeros(len(reordered_cluster_indices), dtype=torch.long, device=cluster_indices.device)
        for new_idx, orig_idx in enumerate(reordered_cluster_indices):
            cluster_priority[orig_idx] = new_idx

        # 할당된 클러스터 ID를 우선순위 값으로 변환
        channel_priority = cluster_priority[cluster_indices]

        # 우선순위 값으로 정렬 (낮은 값이 먼저 오도록)
        _, sorted_indices = torch.sort(channel_priority)

        # 채널 순서 재배열
        inputs = inputs[:, :, sorted_indices]  # Re-order channels based on cluster priority

        # Cluster-wise convolution
        outputs = []
        for idx in reordered_cluster_indices:
            conv_layer = self.convs[idx]
            inputs_permuted = inputs.permute(0, 2, 1).unsqueeze(1)
            conv_out = conv_layer(inputs_permuted)
            outputs.append(conv_out)
        outputs = torch.stack(outputs, dim=-1)  # B, output_size, C, L, n_cluster
        outputs = torch.einsum('btclk,ck->btcl', outputs, cluster_prob)  # B, T, C, L
        outputs = outputs.permute(1, 0, 2, 3)  # T, B, C, L
        spks = self.lif(outputs)  # T, B, C, L
        return spks, cluster_prob

    def reorder_clusters_by_distance(self, emb_distance):
        """
        클러스터 임베딩 간 거리를 기반으로 클러스터를 재정렬합니다.
        
        Args:
            emb_distance (torch.Tensor): [K, K] 크기의 클러스터 간 거리 행렬
        
        Returns:
            torch.Tensor: 재정렬된 클러스터 인덱스 (ex: [2, 0, 1])
        """
        K = emb_distance.size(0)
        
        # 거리 행렬에서 모든 쌍의 거리 추출 (대각선 제외)
        pairs = []
        for i in range(K):
            for j in range(i+1, K):  # 중복 제거 (거리 행렬은 대칭)
                pairs.append((i, j, emb_distance[i, j].item()))
        
        # 거리 기준으로 오름차순 정렬
        pairs.sort(key=lambda x: x[2])
        
        # 클러스터 순서 결정 (그리디 방식)
        ordered_clusters = []
        
        # 처음 두 클러스터는 가장 가까운 쌍
        first_pair = pairs[0]
        ordered_clusters.extend([first_pair[0], first_pair[1]])
        
        # 남은 클러스터들을 현재 집합에 가장 가까운 순서대로 추가
        remaining = set(range(K)) - set(ordered_clusters)
        
        while remaining:
            # 현재까지의 집합과 가장 가까운 남은 클러스터 찾기
            best_dist = float('inf')
            best_cluster = -1
            
            for cluster in remaining:
                # 이미 선택된 클러스터들과의 평균 거리 계산
                avg_dist = sum(emb_distance[cluster, c].item() for c in ordered_clusters) / len(ordered_clusters)
                
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_cluster = cluster
            
            ordered_clusters.append(best_cluster)
            remaining.remove(best_cluster)
        
        return torch.tensor(ordered_clusters, device=emb_distance.device)

def channel_shuffle(inputs, shuffle_dim=2):
    '''
    Input
    - inputs: T, B, C, L
    - shuffle_dim: Dimension to shuffle (default: 2 for C channel dimension)

    Return
    - shuffled_inputs: T, B, C, L with channels shuffled along the specified dimension
    '''
    print(f"Shuffling inputs along dimension {shuffle_dim}...")

    if not isinstance(inputs, torch.Tensor):
        raise ValueError("inputs must be a torch.Tensor.")
    
    T, B, C, L = inputs.size()
    
    if shuffle_dim == 2:  # Shuffle channels (C dimension)
        # Generate random permutation indices for channels
        perm_indices = torch.randperm(C, device=inputs.device)
        # Apply shuffling to channel dimension
        shuffled_inputs = inputs[:, :, perm_indices, :]
        
    elif shuffle_dim == 1:  # Shuffle batch dimension
        # Generate random permutation indices for batch
        perm_indices = torch.randperm(B, device=inputs.device)
        # Apply shuffling to batch dimension
        shuffled_inputs = inputs[:, perm_indices, :, :]
        
    elif shuffle_dim == 0:  # Shuffle time dimension
        # Generate random permutation indices for time
        perm_indices = torch.randperm(T, device=inputs.device)
        # Apply shuffling to time dimension
        shuffled_inputs = inputs[perm_indices, :, :, :]
        
    elif shuffle_dim == 3:  # Shuffle sequence length dimension
        # Generate random permutation indices for sequence length
        perm_indices = torch.randperm(L, device=inputs.device)
        # Apply shuffling to sequence length dimension
        shuffled_inputs = inputs[:, :, :, perm_indices]
        
    else:
        raise ValueError(f"Invalid shuffle_dim: {shuffle_dim}. Must be 0, 1, 2, or 3.")
    
    return shuffled_inputs
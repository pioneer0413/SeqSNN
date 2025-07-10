import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True

use_temporal_jitter = False
use_channel_shuffle = False

class AllZerosEncoder(nn.Module):
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
        inputs = torch.zeros(
            (self.out_size, *inputs.size()), dtype=inputs.dtype, device=inputs.device
        )  # T B L C
        inputs = inputs.permute(0, 1, 3, 2)  # T B C L
        #spks = self.lif(inputs)  # T B C L
        return inputs  # No spikes, just zeros
    
class AllOnesEncoder(nn.Module):
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
        inputs = torch.ones(
            (self.out_size, *inputs.size()), dtype=inputs.dtype, device=inputs.device
        )  # T B L C
        inputs = inputs.permute(0, 1, 3, 2)  # T B C L
        #spks = self.lif(inputs)  # T B C L
        return inputs  # No spikes, just ones
    
class RandomEncoder(nn.Module):
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
        B, L, C = inputs.shape
        T = self.out_size
        inputs = torch.randint(0, 2, (T, B, C, L), dtype=inputs.dtype, device=inputs.device)  # T B L C
        return inputs  # No spikes, just random values

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

        if use_temporal_jitter:
            spks = temporal_jitter(spks, only_positive=True, scale='channel', jitter_scale=2)
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

        if use_temporal_jitter:
            spks = temporal_jitter(spks, only_positive=True, scale='channel', jitter_scale=2)
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

        if use_temporal_jitter:
            spks = temporal_jitter(spks, only_positive=True, scale='channel', jitter_scale=2)
        if use_channel_shuffle:
            spks = channel_shuffle(spks, shuffle_dim=0)  # Shuffle channels
        return spks

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

def temporal_jitter(inputs, only_positive=True, scale='all', jitter_scale: int=1):
    '''
    Input
    - inputs: T, B, C, L
    - only_positive: Jitter only toward the postive(=future) direction
    - scale: 'all', 'channel', 'sequence'
      - all: Applying same jitter to all channels
      - channel: Applying different jitter to each channel
      - sequence: Applying different jitter to each sequence
    - jitter_scale: Integer, jitter indicates the number of positions to shift along L axis
    
    Return
    - jittered_inputs: T, B, C, L
    '''

    if not isinstance(jitter_scale, int):
        raise ValueError("jitter_scale must be an integer.")
    
    T, B, C, L = inputs.size()
    
    jittered_inputs = torch.zeros_like(inputs, dtype=inputs.dtype, device=inputs.device)

    if scale == 'all':
        jittered_inputs = torch.roll(inputs, shifts=jitter_scale, dims=-1)  # Shift all channels and sequences
        jittered_inputs = jittered_inputs.reshape(T, B, C, L)  # Reshape to original shape

    elif scale == 'channel': 
        for c in range(C):
            if only_positive:
                jittered_inputs[:, :, c, :] = torch.roll(inputs[:, :, c, :], shifts=jitter_scale, dims=-1)
            else:
                prob = torch.rand(1)
                if prob < 0.5:
                    jitter_scale *= -1  # Shift backward
                jittered_inputs[:, :, c, :] = torch.roll(inputs[:, :, c, :], shifts=jitter_scale, dims=-1)

    elif scale == 'sequence':
        reshaped_inputs = inputs.reshape(T*B*C, L)

        # TODO: Implement different jitter for each sequence

        # recover the original shape
        jittered_inputs = reshaped_inputs.reshape(T, B, C, L)
    
    return jittered_inputs
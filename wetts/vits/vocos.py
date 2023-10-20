import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torchaudio.transforms import InverseSpectrogram


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(1, channels, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x


class ConvNeXtLayer(nn.Module):
    def __init__(self, channels, h_channels, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding="same",
            groups=channels,  # origian is 7
        )
        self.norm = LayerNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1)
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1)
        self.scale = nn.Parameter(
            torch.full(size=(1, channels, 1), fill_value=scale), requires_grad=True
        )

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x


class VocosGenerator(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        h_channels,
        out_channels,
        num_layers,
        istft_config,
        gin_channels,
    ):
        self.in_channels = in_channels
        self.channels = channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.istft_config = istft_config
        self.gin_channels = gin_channels

        super().__init__()

        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=1, padding="same")
        self.cond = Conv1d(gin_channels, channels, 1)
        self.norm_pre = LayerNorm(channels)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channels, h_channels, scale) for _ in range(num_layers)]
        )
        self.norm_post = LayerNorm(channels)
        self.out_conv = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.istft = InverseSpectrogram(**istft_config)

    def forward(self, x, g=None):
        x = self.pad(x)
        x = self.in_conv(x) + self.cond(g)
        x = self.norm_pre(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_post(x)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o

    def remove_weight_norm(self):
        pass

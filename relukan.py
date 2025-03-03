import torch
import torch.nn.functional as F
from torch import nn

from pytorch_nndct.utils import register_custom_op

@register_custom_op("subtract")
def subtract(ctx, a, b):
    return a - b

class ReLUKANConv2DLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        g: int = 4,
        k: int = 2,
        padding=0,
        stride=1,
        train_ab: bool = True,
    ):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.g = g
        self.k = k
        self.r = 4 * g * g / ((k + 1) * (k + 1))
        self.train_ab = train_ab
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.base_activation = nn.ReLU(inplace=True)

        self.base_conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, groups=1, bias=False)
        self.relukan_conv = nn.Conv2d((self.g + self.k) * input_dim, output_dim, kernel_size, stride, padding, groups=1, bias=False)

        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        phase_dims = (1, 1, 1, input_dim, k + g)
        # phase_dims = (1, input_dim, k + g, 1, 1)

        self.phase_low = nn.Parameter((phase_low[None, :].expand(input_dim, -1)).view(*phase_dims), requires_grad=train_ab)
        self.phase_high = nn.Parameter((phase_high[None, :].expand(input_dim, -1)).view(*phase_dims), requires_grad=train_ab)
        self.layer_norm = nn.BatchNorm2d(output_dim)

        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.relukan_conv.weight, nonlinearity='linear')

    def forward(self, x):
        basis = self.base_conv(self.base_activation(x))

        x = x.permute(0, 2, 3, 1)
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], s[3], 1)
        # x = x.view(s[0], s[1], 1, s[2], s[3])

        x1 = F.relu(subtract(x, self.phase_low), inplace=True)
        x2 = F.relu(subtract(self.phase_high, x), inplace=True)
        x = x1 * x2 * self.r
        x = x * x

        s = x.shape
        # x = x.view(s[0], s[1] * s[2], s[3], s[4])
        x = x.reshape(s[0], s[1], s[2], s[3] * s[4])
        x = x.permute(0, 3, 1, 2)

        y = self.relukan_conv(x)

        y = self.base_activation(self.layer_norm(y + basis))
        return y

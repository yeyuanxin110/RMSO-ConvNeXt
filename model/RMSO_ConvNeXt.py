import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        out = x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0., linear=False):
        super().__init__()
        self.linear = linear
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=7, groups=32, padding=3)
        self.act = nn.GELU()
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format='channels_first')
        if not self.linear:
            self.pwconv1 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
            self.pwconv2 = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)
            self.grn = GlobalResponseNorm(4 * out_channels, channels_last=False)
        else:
            self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)
            self.pwconv2 = nn.Linear(4 * out_channels, out_channels)
            self.grn = GlobalResponseNorm(4 * out_channels, channels_last=True)
        self.drop_path = DropPath(drop_path)
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          bias=False))

    def forward(self, x):
        input = self.shortcut(x)
        x = self.dwconv(x)
        x = self.norm(x)

        if not self.linear:
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.grn(x)
            x = self.pwconv2(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.grn(x)
            x = self.pwconv2(x)
            x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class SOConvNeXt(nn.Module):

    def __init__(self,
                 depths=[1, 1, 1, 1],
                 dims=[32, 32, 64, 32],
                 num_features=8,
                 in_chans=1,
                 drop_path_rate=0.):
        super().__init__()
        self.depths = depths

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.Sequential(
            *[Block(in_channels=dims[0], out_channels=dims[0 + 1], drop_path=dp_rates[cur + j]) for j in
              range(depths[0])])
        cur += depths[0]

        self.stage2 = nn.Sequential(
            *[Block(in_channels=dims[1], out_channels=dims[1 + 1], drop_path=dp_rates[cur + j]) for j in
              range(depths[1])])
        cur += depths[1]

        self.stage3 = nn.Sequential(
            *[Block(in_channels=dims[2] + dims[0], out_channels=dims[2 + 1], drop_path=dp_rates[cur + j]) for j in
              range(depths[2])])
        cur += depths[2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=dims[-1] + dims[1], out_channels=num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
        )
        self.aspp = ASPP(64, atrous_rates=[1, 2, 3])

    def forward(self, x):
        x0 = self.conv1(x)

        x1 = self.stage1(x0)
        x2 = torch.cat([self.stage2(x1), x0], 1)

        x3 = torch.cat([self.stage3(x2), x1], 1)

        output = self.aspp(x3)

        return output


class L2NormDense(nn.Module):

    def __init__(self):
        super(L2NormDense, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)

        return x


class RMSO_ConvNeXt(nn.Module):
    """RMSO_ConvNeXt model definition
    """
    def __init__(self, num_features=8):
        super(RMSO_ConvNeXt, self).__init__()
        self.ResNet_Opt = SOConvNeXt(num_features=num_features)
        self.ResNet_Sar = SOConvNeXt(num_features=num_features)

    def input_norm(self, x):
        flat = x.contiguous().view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
                x -
                mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input_opt, input_sar, input_sar_mask=None):
        input_opt = self.input_norm(input_opt)
        input_sar = self.input_norm(input_sar)
        features_opt = self.ResNet_Opt(input_opt)
        features_sar = self.ResNet_Sar(input_sar)
        if input_sar_mask is None:
            return L2NormDense()(features_opt), L2NormDense()(features_sar)
        else:
            input_sar_mask = self.input_norm(input_sar_mask)
            features_sar_mask = self.ResNet_Sar(input_sar_mask)
            return L2NormDense()(features_opt), L2NormDense()(features_sar), L2NormDense()(features_sar_mask)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=8):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange, Reduce


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            act_name="relu",
            is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class SIU(nn.Module):
    def __init__(self, in_dim_l, in_dim_m, in_dim_s):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim_l, in_dim_l, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim_l, in_dim_m, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim_m, in_dim_m, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim_s, in_dim_m, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim_m, in_dim_m, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim_m, in_dim_m, 1),
            # ConvBNReLU(in_dim_m, in_dim_m, 3, 1, 1),
            ConvBNReLU(in_dim_m, in_dim_l, 3, 1, 1),
            nn.Conv2d(in_dim_l, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = F.interpolate(s, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s  # in_dim_m 192
        # lms = torch.cat([attn_l * l, attn_m * m, attn_s * s], dim=1)   # 192*3=576

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class SIU_out(nn.Module):
    def __init__(self, in_dim_l, in_dim_m, in_dim_s, out_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim_l, in_dim_l, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim_l, out_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim_m, out_dim, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim_s, in_dim_m, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim_m, out_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * out_dim, in_dim_m, 1),
            # ConvBNReLU(in_dim_m, in_dim_m, 3, 1, 1),
            ConvBNReLU(in_dim_m, in_dim_l, 3, 1, 1),
            nn.Conv2d(in_dim_l, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = F.interpolate(s, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s  # in_dim_m 192
        # lms = torch.cat([attn_l * l, attn_m * m, attn_s * s], dim=1)   # 192*3=576

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class TwoScaleMix(nn.Module):  # 192-512-1024
    def __init__(self, in_dim_l, in_dim_m, in_dim_s, out_dim):
        super().__init__()
        self.up_sample = nn.Sequential(Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                                       nn.LayerNorm(in_dim_s // 4),
                                       )  # 96
        self.channel_proj = nn.Sequential(nn.Linear(in_dim_m + in_dim_s // 4, 512),
                                          nn.Linear(512, 1024),
                                          nn.LayerNorm(1024)
                                          )
        self.input_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, xs):
        m, s = xs[1].decompose()[0], xs[2].decompose()[0]
        s = self.up_sample(s.permute(0, 2, 3, 1))  # 384-96
        out = torch.cat([m.permute(0, 2, 3, 1), s], dim=3)  # 192+96=288
        backbone_out = self.channel_proj(out).permute(0, 3, 1, 2)
        out = self.input_proj(backbone_out)  # 288
        return out


class ThreeScaleMix(nn.Module):  # 192-512-1024
    def __init__(self, in_dim_l, in_dim_m, in_dim_s, out_dim):
        super().__init__()
        self.down_sample = nn.Sequential(Rearrange('b c (h neih) (w neiw) -> b (neiw neih c) h w', neih=2, neiw=2),
                                         nn.Conv2d(in_dim_l * 4, in_dim_l, 1),
                                         nn.BatchNorm2d(in_dim_l),
                                         nn.ReLU(inplace=True),
                                         )
        self.up_sample = nn.Sequential(Rearrange('b (neiw neih c) h w  -> b c (h neih) (w neiw)', neih=2, neiw=2),
                                       nn.BatchNorm2d(in_dim_s // 4),
                                       )  # 96
        self.channel_proj = nn.Sequential(nn.Conv2d(in_dim_m + in_dim_s // 4 + in_dim_l, 512, 1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(512, 1024, 1),
                                          nn.BatchNorm2d(1024),
                                          )
        self.input_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, xs):
        l, m, s = xs[0].decompose()[0], xs[1].decompose()[0], xs[2].decompose()[0]
        l = self.down_sample(l)
        s = self.up_sample(s)  # 384-96
        out = torch.cat([l, m, s], dim=1)  # 192+96+96=384
        out = self.input_proj(self.channel_proj(out))  # 384-1024-256
        return out


class ThreeScaleMix_672(nn.Module):  # 192-512-1024
    def __init__(self, in_dim_l, in_dim_m, in_dim_s, out_dim):
        super().__init__()
        self.down_sample = nn.Sequential(Rearrange('b c (h neih) (w neiw) -> b (neiw neih c) h w', neih=2, neiw=2),
                                         nn.Conv2d(in_dim_l * 4, in_dim_l, 1),
                                         nn.BatchNorm2d(in_dim_l),
                                         nn.ReLU(inplace=True),
                                         )
        self.channel_proj = nn.Sequential(nn.Conv2d(in_dim_m + in_dim_s + in_dim_l, 1024, 1),
                                          nn.BatchNorm2d(1024),
                                          nn.ReLU(inplace=True),
                                          )
        self.input_proj = nn.Conv2d(1024, out_dim, kernel_size=1)

    def forward(self, xs):
        l, m, s = xs[0].decompose()[0], xs[1].decompose()[0], xs[2].decompose()[0]
        l = self.down_sample(l)
        s = F.interpolate(s, scale_factor=(2, 2), mode='bilinear', align_corners=False)  # 384
        out = torch.cat([l, m, s], dim=1)  # 96+192+384=672
        out = self.input_proj(self.channel_proj(out))  # 384-1024-256
        return out


if __name__ == '__main__':
    net = SIU(96, 192, 384)
    # print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('n_parameters:', n_parameters)

    net2 = SIU_out(96, 192, 384, 384)
    # print(net2)
    n_parameters = sum(p.numel() for p in net2.parameters() if p.requires_grad)
    print('n_parameters:', n_parameters)

    net3 = TwoScaleMix(96, 192, 384, 256)
    print(net3)
    n_parameters = sum(p.numel() for p in net3.parameters() if p.requires_grad)
    print('n_parameters:', n_parameters)
    net4 = ThreeScaleMix(96, 192, 384, 256)
    # print(net4)
    n_parameters = sum(p.numel() for p in net4.parameters() if p.requires_grad)
    print('n_parameters:', n_parameters)
    net5 = ThreeScaleMix_672(96, 192, 384, 256)
    # print(net5)
    n_parameters = sum(p.numel() for p in net5.parameters() if p.requires_grad)
    print('n_parameters:', n_parameters)

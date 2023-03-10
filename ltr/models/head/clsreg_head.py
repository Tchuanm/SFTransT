from typing import Optional
from torch import nn, Tensor
from ltr.models.neck.past_neck.transformer_encoder import _get_activation_fn  #


def conv3x3(input_dim, hidden_dim, output_dim, layer=3):
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True),
    )


class ConvFFN3(nn.Module):  # 256-256
    def __init__(self, input_dim, hidden_dim, output_dim, layer=3):
        super().__init__()
        assert layer == 3, 'layer shoud be 3.'
        self.conv = nn.Sequential(conv3x3(input_dim, hidden_dim),
                                  conv3x3(hidden_dim, hidden_dim),
                                  conv3x3(hidden_dim, output_dim)
                                  )

    def forward(self, x):  # x: 1,B, 1024 ,256(C)
        return self.conv(x)


class ConvFFN(nn.Module):  # 1x1 ffn
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, layer=None):
        super().__init__()
        if layer == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
            )
        elif layer == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
            )
        elif layer == 1:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=True)
        else:
            raise ValueError("invalid input for layer: {} (type: {}, should be in 1,2,3)".format(layer, type(layer)))

    def forward(self, x):  # x: 1,B, 1024 ,256(C)
        return self.conv(x)  # .reshape(x.shape[0], 1024, -1)


class CLS_REG(nn.Module):  # MHA+2FC(256-32-2or4)
    def __init__(self, input_dim, output_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, 8, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 4, input_dim // 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 16, output_dim, kernel_size=1),
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src_search, mask_search, pos_search):
        src_search = src_search.squeeze(0).permute(1, 0, 2)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_search = mask_search.flatten(1)
        # MHA
        tgt2 = self.multihead_attn(query=self.with_pos_embed(src_search, pos_search),
                                   key=self.with_pos_embed(src_search, pos_search),
                                   value=src_search, attn_mask=None,
                                   key_padding_mask=mask_search)[0]
        tgt = src_search + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # conv1x1 X 3layer
        tgt = tgt.unsqueeze(0).permute(1, 3, 0, 2)
        tgt = self.head(tgt)
        tgt = tgt.permute(2, 0, 3, 1).squeeze(0)
        return tgt


class CLS_REG2(nn.Module):  # MHA+2FC+3Conv(1x1)
    def __init__(self, input_dim, output_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, 8, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, input_dim * 8)
        self.linear2 = nn.Linear(input_dim * 8, input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 4, input_dim // 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 16, output_dim, kernel_size=1),
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src_search, mask_search, pos_search):
        src_search = src_search.squeeze(0).permute(1, 0, 2)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_search = mask_search.flatten(1)
        # MHA for FFN(256-64-16-4(2))
        tgt2 = self.multihead_attn(query=self.with_pos_embed(src_search, pos_search),
                                   key=self.with_pos_embed(src_search, pos_search),
                                   value=src_search, attn_mask=None,
                                   key_padding_mask=mask_search)[0]
        tgt = src_search + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)

        tgt = tgt.unsqueeze(0).permute(1, 3, 0, 2)
        tgt = self.head(tgt)
        tgt = tgt.permute(2, 0, 3, 1).squeeze(0)
        return tgt


# cls,reg: 5layer:7layers
class ClsHead(nn.Module):  # 6Conv(1x1)
    # 256--64-32-16-8-2
    def __init__(self, input_dim, output_dim, num_layers=6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 4, input_dim // 8, kernel_size=1),
            nn.BatchNorm2d(input_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 8, input_dim // 16, kernel_size=1),
            nn.BatchNorm2d(input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 16, output_dim, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(2, 3, 0, 1)
        x = self.head(x)
        x = x.permute(2, 3, 0, 1)
        return x


class RegHead(nn.Module):  # 8 Conv(1x1)
    # 256-256-128-64-32-16-8-4     1222222 7layer
    def __init__(self, input_dim, output_dim, num_layers=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),  ###########
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False),  ###########
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 4, input_dim // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 8, input_dim // 16, kernel_size=1),
            nn.BatchNorm2d(input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 16, input_dim // 32, kernel_size=1),
            nn.BatchNorm2d(input_dim // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 32, output_dim, kernel_size=1),  ######
        )

    def forward(self, x):
        x = x.permute(2, 3, 0, 1)
        x = self.head(x)
        x = x.permute(2, 3, 0, 1)
        return x

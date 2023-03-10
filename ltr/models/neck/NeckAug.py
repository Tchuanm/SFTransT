"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .gaussian_hfe_layer import GaussianMultiheadAttention
from ltr.models.neck.misc import MLP


class NeckAugument(nn.Module):
    def __init__(self, d_model=256, nhead=8, iteration=4, iteration_gpha=4, dim_feedforward=2048, dropout=0.1,
                 out_size=32, activation="relu", return_intermediate_dec=True, smooth=8, dynamic_scale='type3'):
        super().__init__()

        # encoer: MHCA*2branch
        encoder_norm = nn.LayerNorm(d_model)
        encoder_layer = CrossAug(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(encoder_layer, iteration, encoder_norm)
        # decoder: GPHA attention x L layers   (kenel part)
        decoder_layers = []
        for layer_index in range(iteration_gpha):
            decoder_layer = TransformerDecoderLayer(dynamic_scale, smooth, layer_index,
                                                    d_model, nhead, dim_feedforward, dropout, out_size,
                                                    activation)
            decoder_layers.append(decoder_layer)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        if dynamic_scale in ["type2", "type3", "type4"]:
            for layer_index in range(iteration_gpha):
                nn.init.zeros_(self.decoder.layers[layer_index].point3.weight)
                with torch.no_grad():
                    nn.init.ones_(self.decoder.layers[layer_index].point3.bias)

        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, h_w,
                src_temp, mask_temp, pos_temp):
        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        bs, c, h, w = src_temp.shape        # for 1024 channel
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(src.device)
        grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * 8, 1)
        # search
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # template
        query_embed = pos_temp.flatten(2).permute(2, 0, 1)
        tgt = src_temp.flatten(2).permute(2, 0, 1)
        mask_tgt = mask_temp.flatten(1)

        tgt, memory = self.encoder(src1=tgt, src2=src,
                                       src1_key_padding_mask=mask_tgt,
                                       src2_key_padding_mask=mask,
                                       pos_src1=query_embed,
                                       pos_src2=pos_embed)
        # new try get 1024 channel  siamft: h_w==[1024,1024]  0.70x
        hs, points = self.decoder(grid, h_w, memory, tgt, memory_key_padding_mask=mask_tgt,
                                  tgt_key_padding_mask=mask,
                                  pos=query_embed, query_pos=pos_embed)
        return hs.transpose(1, 2), points.transpose(0, 1)


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = norm
        self.norm2 = copy.deepcopy(norm)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        if self.norm1 is not None:
            output1 = self.norm1(src1)
            output2 = self.norm2(src2)

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)
        return output1, output2


class CrossAug(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # branch1
        self.sa_qcontent_proj1 = nn.Linear(d_model, d_model)
        self.sa_qpos_proj1 = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj1 = nn.Linear(d_model, d_model)
        self.sa_kpos_proj1 = nn.Linear(d_model, d_model)
        self.sa_v_proj1 = nn.Linear(d_model, d_model)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.dropout13 = nn.Dropout(dropout)
        self.norm13 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(dim_feedforward, d_model),)
        self.dropout14 = nn.Dropout(dropout)
        self.norm14 = nn.LayerNorm(d_model)
        # branch2
        self.sa_qcontent_proj2 = nn.Linear(d_model, d_model)
        self.sa_qpos_proj2 = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj2 = nn.Linear(d_model, d_model)
        self.sa_kpos_proj2 = nn.Linear(d_model, d_model)
        self.sa_v_proj2 = nn.Linear(d_model, d_model)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.dropout23 = nn.Dropout(dropout)
        self.norm23 = nn.LayerNorm(d_model)

        self.MLP2 = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(dim_feedforward, d_model),)
        self.dropout24 = nn.Dropout(dropout)
        self.norm24 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        #  branch1 src1=template; src2=search
        q_content = self.sa_qcontent_proj1(src1)
        q_pos = self.sa_qpos_proj1(pos_src1)
        k_content = self.sa_kcontent_proj1(src2)
        k_pos = self.sa_kpos_proj1(pos_src2)
        v = self.sa_v_proj1(src2)

        src13 = self.multihead_attn1(q_content + q_pos, k_content + k_pos, value=v,
                                     attn_mask=src2_mask, key_padding_mask=src2_key_padding_mask)[0]
        src1 = src1 + self.dropout13(src13)
        src1 = self.norm13(src1)

        src13 = self.MLP1(src1)          # MLP
        src1 = src1 + self.dropout14(src13)
        src1 = self.norm14(src1)

        # branch2
        q_content = self.sa_qcontent_proj2(src2)
        q_pos = self.sa_qpos_proj2(pos_src2)
        k_content = self.sa_kcontent_proj2(src1)
        k_pos = self.sa_kpos_proj2(pos_src1)
        v = self.sa_v_proj2(src1)

        src23 = self.multihead_attn2(q_content + q_pos, k_content + k_pos, value=v,
                                     attn_mask=src1_mask, key_padding_mask=src1_key_padding_mask)[0]
        src2 = src2 + self.dropout23(src23)
        src2 = self.norm23(src2)

        src23 = self.MLP2(src2)           # MLP
        src2 = src2 + self.dropout24(src23)
        src2 = self.norm24(src2)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, norm, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layer)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, grid, h_w, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        points = []
        point_sigmoid_ref = None
        for layer in self.layers:
            output, point, point_sigmoid_ref = layer(
                grid, h_w, output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos, point_ref_previous=point_sigmoid_ref
            )
            points.append(point)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), points[0]

        return output.unsqueeze(0), points[0]


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dynamic_scale, smooth, layer_index,
                 d_model, nhead, dim_feedforward=2048, dropout=0.1, out_size=32,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.smooth = smooth
        self.dynamic_scale = dynamic_scale
        self.out_size = out_size

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if layer_index == 0:
            self.point1 = MLP(256, 256, 2, 3)
            self.point2 = nn.Linear(d_model, 2 * 8)
        else:
            self.point2 = nn.Linear(d_model, 2 * 8)
        self.layer_index = layer_index
        if self.dynamic_scale == "type2":
            self.point3 = nn.Linear(d_model, 8)
        elif self.dynamic_scale == "type3":
            self.point3 = nn.Linear(d_model, 16)
        elif self.dynamic_scale == "type4":
            self.point3 = nn.Linear(d_model, 24)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, grid, h_w, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     point_ref_previous: Optional[Tensor] = None):
        tgt_len = tgt.shape[0]

        out = self.norm4(tgt + query_pos)
        point_sigmoid_offset = self.point2(out)
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.layer_index == 0:
            point_sigmoid_ref_inter = self.point1(out)
            point_sigmoid_ref = point_sigmoid_ref_inter.sigmoid()
            point_sigmoid_ref = (h_w - 0) * point_sigmoid_ref / self.out_size
            point_sigmoid_ref = point_sigmoid_ref.repeat(1, 1, 8)
        else:
            point_sigmoid_ref = point_ref_previous

        point = point_sigmoid_ref + point_sigmoid_offset
        point = point.view(tgt_len, -1, 2)
        distance = (point.unsqueeze(1) - grid.unsqueeze(0)).pow(2)

        if self.dynamic_scale == "type1":
            scale = 1
            distance = distance.sum(-1) * scale
        elif self.dynamic_scale == "type2":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1).unsqueeze(1)
            distance = distance.sum(-1) * scale
        elif self.dynamic_scale == "type3":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1, 2).unsqueeze(1)
            distance = (distance * scale).sum(-1)
        elif self.dynamic_scale == "type4":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1, 3).unsqueeze(1)
            distance = torch.cat([distance, torch.prod(distance, dim=-1, keepdim=True)], dim=-1)
            distance = (distance * scale).sum(-1)

        gaussian = -(distance - 0).abs() / self.smooth

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=[gaussian])[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.layer_index == 0:
            return tgt, point_sigmoid_ref_inter, point_sigmoid_ref
        else:
            return tgt, None, point_sigmoid_ref

    def forward(self, grid, h_w, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                point_ref_previous: Optional[Tensor] = None):

        return self.forward_post(grid, h_w, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 point_ref_previous)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_neck(settings):
    return NeckAugument(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        iteration=settings.iteration,
        iteration_gpha=settings.iteration_gpha,
        activation=settings.activation,
        out_size=settings.search_feature_sz,
        # return_intermediate_dec=settings.return_intermediate,
        smooth=8,
        dynamic_scale='type3',
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    # import ltr.admin.settings as ws_settings
    # settings = ws_settings.Settings()
    # settings.hidden_dim = 2048
    # settings.dropout = 0.1
    neck_net = NeckAugument()
    print(neck_net)

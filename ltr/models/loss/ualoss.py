import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_coef(iter_percentage=None, method='cos'):
    if method == 'cos':
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    elif method == 'constant':
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits):       # pred; gtbox
    # assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    # if seg_logits.shape != seg_gts.shape:
    #     seg_logits = seg_logits[:, 0, :]

    # sigmoid_b = seg_logits[:, 0, :].sigmoid()   # background sigmoid prob
    # loss_map = 1 - (2 * sigmoid_x - 1).pow(2)
    sigmoid_f = seg_logits[:, 1, :].sigmoid()   # forground sigmoid prob
    loss_map = 1 - (2 * sigmoid_f - 1).pow(2)       # + (2 * sigmoid_f - 1).pow(2))
    ual_loss = loss_map.mean()
    # ual_coef = get_coef(method='constant')  # weight == 1
    # ual_loss = ual_loss * ual_coef
    return ual_loss



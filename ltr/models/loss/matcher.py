# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn
import numpy as np


class TrackingMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]

        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item();
            cy = cy.item();
            w = w.item();
            h = h.item()
            xmin = cx - w / 2;
            ymin = cy - h / 2;
            xmax = cx + w / 2;
            ymax = cy + h / 2
            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin * len_feature))
            Ymin = int(np.ceil(ymin * len_feature))
            Xmax = int(np.ceil(xmax * len_feature))
            Ymax = int(np.ceil(ymax * len_feature))
            if Xmin == Xmax:
                Xmax = Xmax + 1
            if Ymin == Ymax:
                Ymax = Ymax + 1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax, Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def generate_points(stride, size):
    x, y = np.meshgrid([stride * dx for dx in np.arange(0, size)],
                       [stride * dy for dy in np.arange(0, size)])
    points = np.zeros((2, size, size), dtype=np.float32)
    points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)
    return points


class MaskTrackingMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = generate_points(1, 32)

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]

        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item();
            cy = cy.item();
            w = w.item();
            h = h.item()
            len_feature = int(np.sqrt(num_queries))
            tcx = cx * len_feature;
            tcy = cy * len_feature;
            tw = w * len_feature;
            th = h * len_feature
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            pos_idx = np.where(np.abs(tcx - self.points[0]) / np.abs(tw / 2) +
                               np.abs(tcy - self.points[1]) / np.abs(th / 2) < 1)
            # Rhombus  /2  eos_coef=0.03125    or *0.6 (TusiamBAN) eos_coef=0.045
            c = b[pos_idx].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class EllipseMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = generate_points(1, 32)

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]

        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item();
            cy = cy.item();
            w = w.item();
            h = h.item()
            len_feature = int(np.sqrt(num_queries))
            tcx = cx * len_feature;
            tcy = cy * len_feature;
            tw = w * len_feature;
            th = h * len_feature
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            pos_idx = np.where(np.square(tcx - self.points[0]) / np.square(tw / 2 + 1e-6) +
                               np.square(tcy - self.points[1]) / np.square(th / 2 + 1e-6) < 1) # ellipse labels;eos_coef=0.049
            c = b[pos_idx].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return TrackingMatcher()


def build_rhombus_matcher():
    return MaskTrackingMatcher()


def build_ellipse_matcher():
    return EllipseMatcher()

from ltr import model_constructor

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import copy
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy, inverse_sigmoid)

from ltr.models.backbone.grswin_backbone import build_backbone
from ltr.models.neck.multi_scale_fusion import TwoScaleMix  # multi-scale-fusion/ intergrate
from ltr.models.neck.NeckAug import build_neck
from ltr.models.neck.misc import MLP
from ltr.models.loss.matcher import build_matcher, build_ellipse_matcher


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class sftranst(nn.Module):
    def __init__(self, backbone, transformer, num_classes=1, settings=None):
        """ Initializes the model.
        Parameters:
            backbone: swinT
            transformer:  feature attention transformer module
        """
        super().__init__()
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.scale_fusion = TwoScaleMix(backbone.num_channels//2, backbone.num_channels,
                                        backbone.num_channels*2, out_dim=self.hidden_dim)
        self.backbone = backbone
        self.out_size = settings.search_feature_sz
        self.h_w = torch.tensor([[[self.out_size**2, self.out_size**2]]])     # 1024 or 1600
        # for reg. 256x3; 256-4
        self.reg_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.bbox_embed = nn.Linear(self.hidden_dim, 4)
        # for cls. 256x3; 256-2
        self.cls_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        # init cls_proj reg_proj
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

    def forward(self, search, template):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        # search multi-scale fusion
        feature_search, pos_search = self.backbone(search)
        src_search = self.scale_fusion(feature_search)
        mask_search = feature_search[1].decompose()[1]
        assert mask_search is not None
        # template
        feature_template, pos_template = self.backbone(template)
        src_template = self.scale_fusion(feature_template)
        mask_template = feature_template[1].decompose()[1]
        assert mask_template is not None
        #   neckhs=(bs 1024 256)
        hs, points = self.transformer(src_search, mask_search, pos_search[1], self.h_w.to(mask_search.device),
                                      src_template, mask_template, pos_template[1])
        # simple head
        cls_feat = self.cls_proj(hs[-1])
        reg_feat = self.reg_proj(hs[-1])
        cls_out = self.class_embed(cls_feat)
        outputs_coord = self.bbox_embed(reg_feat)
        outputs_coord[..., :2] = outputs_coord[..., :2] + points.unsqueeze(0)
        out = {'pred_logits': cls_out, 'pred_boxes': outputs_coord.sigmoid()}    # 1,1024,2;  1,1024,4
        return out

    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        feature_search, pos_search = self.backbone(search)
        src_search = self.scale_fusion(feature_search)
        mask_search = feature_search[1].decompose()[1]
        assert mask_search is not None
        hs, points = self.transformer(src_search, mask_search, pos_search[1], self.h_w.to(mask_search.device),
                                      self.src_template, self.mask_template, self.pos_template[1])
        # simple head
        cls_feat = self.cls_proj(hs[-1])
        reg_feat = self.reg_proj(hs[-1])
        cls_out = self.class_embed(cls_feat)
        outputs_coord = self.bbox_embed(reg_feat)
        outputs_coord[..., :2] = outputs_coord[..., :2] + points.unsqueeze(0)
        out = {'pred_logits': cls_out, 'pred_boxes': outputs_coord.sigmoid()}    # 1,1024,2;  1,1024,4
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, self.pos_template = self.backbone(z)
        self.src_template = self.scale_fusion(zf)
        self.mask_template = zf[1].decompose()[1]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # bce loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # ciou loss
        ciou_loss, iou = box_ops.bbox_ciou(src_boxes, target_boxes)  # src: (bs x channel,4)  (target: bs,4)
        # ciou_loss, iou = box_ops.cdiou(src_boxes, target_boxes)  # src: (bs x channel,4)  (target: bs,4)
        losses['loss_ciou'] = ciou_loss
        losses['iou'] = iou
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)
        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses


@model_constructor
def sftranst_network(settings):
    backbone_net = build_backbone(settings)
    transformer = build_neck(settings)
    model = sftranst(backbone_net, transformer, num_classes=1, settings=settings)
    device = torch.device(settings.device)
    model.to(device)
    return model


def losses_combination(settings):
    """loss combination: ce loss ,L1 loss , CIOU loss"""
    matcher = build_ellipse_matcher()
    weight_dict = {'loss_ce': settings.loss_cls_weight, 'loss_bbox': settings.loss_l1_weight,
                   'loss_ciou': settings.loss_iou_weight}   #
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes=1, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0516, losses=losses)          # 0.0625  0.0491
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion

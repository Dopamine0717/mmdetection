# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .builder import MATCH_COST


@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        if self.iou_mode=='giou':
            # overlaps: [num_bboxes, num_gt]
            overlaps = bbox_overlaps(
                bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
            # The 1 is a constant that doesn't change the matching, so omitted.
            iou_cost = -overlaps
        if self.iou_mode=='ciou':  #增加了ciou
            # overlap
            eps=1e-6
            lt = torch.max(bboxes[:, None, :2], gt_bboxes[:, None, :2])
            rb = torch.min(bboxes[:, None, 2:], gt_bboxes[:, None, 2:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[:, 0] * wh[:, 1]

            # union
            ap = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            ag = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            union = ap[..., None] + ag[..., None, :] - overlap + eps

            # IoU
            ious = overlap / union

            # enclose area
            enclose_x1y1 = torch.min(bboxes[:, None, :2], gt_bboxes[:, None, :2])
            enclose_x2y2 = torch.max(bboxes[:, None, 2:], gt_bboxes[:, None, 2:])
            enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

            cw = enclose_wh[:, 0]
            ch = enclose_wh[:, 1]

            c2 = cw**2 + ch**2 + eps

            b1_x1, b1_y1 = bboxes[:, 0], bboxes[:, 1]
            b1_x2, b1_y2 = bboxes[:, 2], bboxes[:, 3]
            b2_x1, b2_y1 = gt_bboxes[:, 0], gt_bboxes[:, 1]
            b2_x2, b2_y2 = gt_bboxes[:, 2], gt_bboxes[:, 3]

            w1, h1 = b1_x2[..., None] - b1_x1[..., None], b1_y2[..., None] - b1_y1[..., None] + eps
            w2, h2 = b2_x2[..., None, :] - b2_x1[..., None, :], b2_y2[..., None, :] - b2_y1[..., None, :] + eps

            left = ((b2_x1[..., None, :] + b2_x2[..., None, :]) - (b1_x1[..., None] + b1_x2[..., None]))**2 / 4
            right = ((b2_y1[..., None, :] + b2_y2[..., None, :]) - (b1_y1[..., None] + b1_y2[..., None]))**2 / 4
            rho2 = left + right
            # 对应公式
            factor = 4 / math.pi**2
            v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

            # CIoU
            cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
            iou_cost=-cious
        return iou_cost * self.weight

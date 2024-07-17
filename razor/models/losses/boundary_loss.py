# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from cucim.skimage.morphology import binary_erosion
from monai.utils import convert_to_cupy, convert_to_tensor


def boundary_pattern_v2(kernel_size=3):
    matrix = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device='cpu')
    matrix = matrix.view(1, 1, kernel_size, kernel_size, kernel_size).cuda()
    return matrix


def get_edges(
    seg_label: torch.Tensor,
    label_idx: int = 1,
) -> torch.Tensor:
    """
    Compute edges from binary segmentation masks. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    Args:
        seg_label: the predicted binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
    """
    converter = partial(convert_to_tensor, device=seg_label.device)

    # If not binary images, convert them
    if seg_label.dtype not in (bool, torch.bool):
        seg_label = seg_label == label_idx

    seg_label = convert_to_cupy(seg_label, dtype=bool)  # type: ignore[arg-type]
    edges_label = binary_erosion(seg_label) ^ seg_label
    return converter(edges_label, dtype=bool)  # type: ignore


class BoundaryKDV1(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 kernel_size: int = 3,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV1, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern_v2(kernel_size)
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        loss = torch.tensor(0.).cuda()
        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(gt_labels[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0)) * (self.temperature**2)
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        return self.loss_weight * loss


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from cucim.skimage.morphology import binary_erosion, binary_dilation
from monai.utils import convert_to_cupy, convert_to_tensor
from .cwd import ChannelWiseDivergence


def get_mask_edges(
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
    # edges_label = binary_erosion(seg_label) ^ seg_label
    edges_label = binary_dilation(seg_label) ^ binary_erosion(seg_label)
    return converter(edges_label, dtype=bool)  # type: ignore


class BPKD(nn.Module):
    def __init__(self,
                 alpha: float = 2.0,
                 temperature: float = 1.0,
                 num_classes: int = 14,
                 edge_weight: float = 50.0,
                 body_weight: float = 20.0,
                 loss_weight: float = 1.0):
        super(BPKD, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.body_weight = body_weight
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size = preds_S.shape[0]
        loss_edges = torch.tensor(0.).cuda()
        loss_bodies = torch.tensor(0.).cuda()
        for bs in range(batch_size):
            loss_edges_i = torch.tensor(0.).cuda()
            # loss_bodies_i = torch.tensor(0.).cuda()
            number_edge_pixels = 0
            number_body_pixels = 0
            for i in range(self.num_classes):
                if i == 0:
                    continue
                mask_labels = gt_labels[bs, 0].detach().clone() == i
                mask_edges_i = get_mask_edges(mask_labels, i)
                mask_bodies_i = ~mask_edges_i * mask_labels
                if mask_edges_i.sum() == 0:
                    continue
                number_edge_pixels += mask_edges_i.sum()
                number_body_pixels += mask_bodies_i.sum()
                preds_edges_S_i = preds_S[bs, i] * mask_edges_i
                preds_edges_T_i = preds_T[bs, i] * mask_edges_i

                preds_bodies_S_i = preds_S[bs, i] * mask_bodies_i
                preds_bodies_T_i = preds_T[bs, i] * mask_bodies_i
                loss_edges_i += F.kl_div(
                    F.log_softmax(preds_edges_S_i.view(-1) / self.temperature, dim=0),
                    F.softmax(preds_edges_T_i.view(-1) / self.temperature, dim=0), reduction='sum') \
                    * (self.temperature ** 2)
                loss_bodies += F.kl_div(
                    F.log_softmax(preds_bodies_S_i.view(-1) / self.temperature, dim=0),
                    F.softmax(preds_bodies_T_i.view(-1) / self.temperature, dim=0), reduction='sum') \
                    * (self.temperature ** 2)
                # img_path = gt_labels.meta['filename_or_obj']
                # affine = gt_labels.meta['original_affine']
                #
                # import numpy as np
                # import os
                # import os.path as osp
                # import nibabel as nib
                #
                # gt_edges_ = mask_edges_i.cpu().numpy().astype(np.uint8)
                # gt_bodies_ = mask_bodies_i.cpu().numpy().astype(np.uint8)
                # gt_labels_ = gt_labels.cpu().numpy().astype(np.uint8)
                #
                # # for bs in range(gt_edges_i.shape[0]):
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{i}_{osp.basename(img_path[bs])}")
                # nib.save(
                #     nib.Nifti1Image(gt_labels_[bs][0].astype(np.uint8), affine[bs]),
                #     save_dirs)
                #
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{i}_{osp.basename(img_path[bs].replace('label', 'gt_edges'))}")
                # nib.save(
                #     nib.Nifti1Image(gt_edges_.astype(np.uint8), affine[bs]),
                #     save_dirs)
                #
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{i}_{osp.basename(img_path[bs].replace('label', 'gt_bodies'))}")
                # nib.save(
                #     nib.Nifti1Image(gt_bodies_.astype(np.uint8), affine[bs]),
                #     save_dirs)
            if loss_edges_i > 0:
                loss_edges += loss_edges_i / number_edge_pixels
            # if number_body_pixels > 0:
            #     loss_bodies += loss_bodies_i / number_body_pixels

        # loss = (self.edge_weight * loss_edges + self.body_weight * loss_bodies / self.num_classes) / batch_size
        loss_edges = self.loss_weight * self.edge_weight * loss_edges / batch_size
        loss_bodies = self.loss_weight * self.body_weight * loss_bodies / (self.num_classes * batch_size)
        return dict(loss_edges=loss_edges, loss_bodies=loss_bodies)
        # return self.loss_weight * loss


class BPKDV2(nn.Module):
    def __init__(self,
                 alpha: float = 2.0,
                 temperature: float = 1.0,
                 num_classes: int = 14,
                 edge_weight: float = 500.0,
                 body_weight: float = 200.0,
                 loss_weight: float = 1.0):
        super(BPKDV2, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.body_weight = body_weight
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size = preds_S.shape[0]
        loss_edges = torch.tensor(0.).cuda()
        loss_bodies = torch.tensor(0.).cuda()
        for bs in range(batch_size):
            loss_edges_i = torch.tensor(0.).cuda()
            loss_bodies_i = torch.tensor(0.).cuda()
            number_edge_pixels = 0
            number_body_pixels = 0
            for i in range(self.num_classes):
                if i == 0:
                    continue
                mask_labels = gt_labels[bs, 0].detach().clone() == i
                mask_edges_i = get_mask_edges(mask_labels, i)
                mask_bodies_i = ~mask_edges_i * mask_labels
                if mask_edges_i.sum() == 0:
                    continue
                number_edge_pixels += mask_edges_i.sum()
                number_body_pixels += mask_bodies_i.sum()
                preds_edges_S_i = preds_S[bs, i] * mask_edges_i
                preds_edges_T_i = preds_T[bs, i] * mask_edges_i

                preds_bodies_S_i = preds_S[bs, i] * mask_bodies_i
                preds_bodies_T_i = preds_T[bs, i] * mask_bodies_i
                loss_edges_i += F.kl_div(
                    F.log_softmax(preds_edges_S_i.view(-1) / self.temperature, dim=0),
                    F.softmax(preds_edges_T_i.view(-1) / self.temperature, dim=0), reduction='sum') \
                    * (self.temperature ** 2)
                loss_bodies_i += F.kl_div(
                    F.log_softmax(preds_bodies_S_i.view(-1) / self.temperature, dim=0),
                    F.softmax(preds_bodies_T_i.view(-1) / self.temperature, dim=0), reduction='sum') \
                    * (self.temperature ** 2)

            if loss_edges_i > 0:
                loss_edges += loss_edges_i / number_edge_pixels
            if number_body_pixels > 0:
                loss_bodies += loss_bodies_i / number_body_pixels

        loss_edges = self.loss_weight * self.edge_weight * loss_edges / batch_size
        loss_bodies = self.loss_weight * self.body_weight * loss_bodies / batch_size
        return dict(loss_edges=loss_edges, loss_bodies=loss_bodies)


class BPKDV3(nn.Module):
    def __init__(self,
                 alpha: float = 2.0,
                 temperature: float = 1.0,
                 num_classes: int = 14,
                 edge_weight: float = 500.0,
                 body_weight: float = 200.0,
                 loss_weight: float = 1.0):
        super(BPKDV3, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.body_weight = body_weight
        self.loss_weight = loss_weight
        self.cwd = ChannelWiseDivergence()

    def forward(self, preds_S, preds_T, gt_labels):

        batch_size = preds_S.shape[0]

        loss_edges = torch.tensor(0.).cuda()
        loss_bodies = torch.tensor(0.).cuda()

        for bs in range(batch_size):

            mask_edges_bs_list = []
            mask_bodies_bs_list = []

            for i in range(self.num_classes):
                if i == 0:
                    mask_edges_bs_list.append(torch.zeros_like(gt_labels)[0])
                    mask_bodies_bs_list.append(torch.zeros_like(gt_labels)[0])
                    continue

                mask_labels = gt_labels[bs, 0].detach().clone() == i

                mask_edges_i = get_mask_edges(mask_labels, i)
                mask_bodies_i = ~mask_edges_i * mask_labels

                mask_edges_bs_list.append(mask_edges_i.unsqueeze(0))
                mask_bodies_bs_list.append(mask_bodies_i.unsqueeze(0))

            mask_edges_bs = torch.concat(mask_edges_bs_list, 0)
            mask_bodies_bs = torch.concat(mask_bodies_bs_list, 0)

            preds_edges_S_i = preds_S[bs] * mask_edges_bs
            preds_edges_T_i = preds_T[bs] * mask_edges_bs

            preds_bodies_S_i = preds_S[bs] * mask_bodies_bs
            preds_bodies_T_i = preds_T[bs] * mask_bodies_bs

            loss_edges += self.cwd(preds_edges_S_i.unsqueeze(0), preds_edges_T_i.unsqueeze(0))
            loss_bodies += self.cwd(preds_bodies_S_i.unsqueeze(0), preds_bodies_T_i.unsqueeze(0))

        loss_edges = self.loss_weight * self.edge_weight * loss_edges / batch_size
        loss_bodies = self.loss_weight * self.body_weight * loss_bodies / batch_size
        return dict(loss_edges=loss_edges, loss_bodies=loss_bodies)


class BPKDV4(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 num_classes: int = 14,
                 edge_weight: float = 500.0,
                 body_weight: float = 200.0,
                 loss_weight: float = 1.0):
        super(BPKDV4, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.body_weight = body_weight
        self.loss_weight = loss_weight
        self.cwd = ChannelWiseDivergence()

    def forward(self, preds_S, preds_T, gt_labels):

        batch_size = preds_S.shape[0]

        mask_edges_list = []
        mask_bodies_list = []

        for bs in range(batch_size):

            mask_edges_bs_list = []
            mask_bodies_bs_list = []

            for i in range(self.num_classes):
                if i == 0:
                    mask_edges_bs_list.append(torch.zeros_like(gt_labels)[0])
                    mask_bodies_bs_list.append(torch.zeros_like(gt_labels)[0])
                    continue

                mask_labels = gt_labels[bs, 0].detach().clone() == i

                mask_edges_i = get_mask_edges(mask_labels, i)
                mask_bodies_i = ~mask_edges_i * mask_labels

                mask_edges_bs_list.append(mask_edges_i.unsqueeze(0))
                mask_bodies_bs_list.append(mask_bodies_i.unsqueeze(0))

            mask_edges_bs = torch.concat(mask_edges_bs_list, 0)
            mask_bodies_bs = torch.concat(mask_bodies_bs_list, 0)

            mask_edges_list.append(mask_edges_bs.unsqueeze(0))
            mask_bodies_list.append(mask_bodies_bs.unsqueeze(0))

        mask_edges = torch.concat(mask_edges_list)
        mask_bodies = torch.concat(mask_bodies_list)

        preds_edges_S_i = preds_S * mask_edges
        preds_edges_T_i = preds_T * mask_edges

        preds_bodies_S_i = preds_S * mask_bodies
        preds_bodies_T_i = preds_T * mask_bodies

        loss_edges = self.cwd(preds_edges_S_i, preds_edges_T_i)
        loss_bodies = self.cwd(preds_bodies_S_i, preds_bodies_T_i)

        loss_edges = self.loss_weight * self.edge_weight * loss_edges / batch_size
        loss_bodies = self.loss_weight * self.body_weight * loss_bodies / batch_size
        return dict(loss_edges=loss_edges, loss_bodies=loss_bodies)


if __name__ == '__main__':
    img_path = 'data/synapse_raw/imagesTr/img0038.nii.gz'
    label_path = img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')
    label = nib.load(label_path).get_fdata()
    gt_labels = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    loss_bpkd = BPKDV3()
    feats_S = torch.rand(1, 14, *gt_labels.shape)
    feats_T = torch.rand(1, 14, *gt_labels.shape)
    loss = loss_bpkd(feats_S, feats_T, gt_labels)
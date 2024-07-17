# -*- coding: utf-8 -*-
# copyright from:
# https://github.com/HiLab-git/LCOVNet-and-KD/blob/ee8ab5e0060d6270abf881a0fcca98f970cfff80/pymic/loss/seg/kd.py
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import Tensor
from torch import nn, Tensor
from monai.networks import one_hot
from mmrazor.models.losses import CrossEntropyLoss
from .dsd import TransConv, init_weights
from mmengine.logging import MessageHub


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


# class MSKDCAKDLoss(nn.Module):
#     def __init__(self, loss_weight, sigmoid=False, cakd_weight=1.0, fnkd_weight=1.0):
#         super(MSKDCAKDLoss, self).__init__()
#         ce_kwargs = {}
#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.loss_weight = loss_weight
#         self.cakd_weight = cakd_weight
#         self.fnkd_weight = fnkd_weight
#         self.sigmoid = sigmoid
#
#     def forward(self, student_outputs, teacher_outputs, student_features, teacher_features):
#         # loss = torch.tensor(0.).cuda()
#         # w = [0.4, 0.2, 0.2, 0.2]
#         # for i in range(0, 4):
#         #     loss += w[i] * (0.1 * self.CAKD(student_outputs[i], teacher_outputs[i])
#         #                     + 0.2 * self.FNKD(student_outputs[i], teacher_outputs[i], student_outputs[i + 4],
#         #                                       teacher_outputs[i + 4]))
#
#         cakd_loss = self.cakd_weight * self.CAKD(student_outputs, teacher_outputs)
#
#         fnkd_loss = self.fnkd_weight * self.FNKD(
#             student_outputs, teacher_outputs, student_features, teacher_features)
#
#         loss = cakd_loss + fnkd_loss
#         return loss * self.loss_weight
#
#     def CAKD(self, student_outputs, teacher_outputs):
#         [B, C, D, W, H] = student_outputs.shape
#
#         if self.sigmoid:
#             student_outputs = torch.sigmoid(student_outputs)
#             teacher_outputs = torch.sigmoid(teacher_outputs)
#         else:
#             student_outputs = F.softmax(student_outputs, dim=1)
#             teacher_outputs = F.softmax(teacher_outputs, dim=1)
#
#         student_outputs = student_outputs.reshape(B, C, D * W * H)
#         teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)
#
#         with autocast(enabled=False):
#             student_outputs = torch.bmm(student_outputs, student_outputs.permute(
#                 0, 2, 1))
#             teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
#                 0, 2, 1))
#         Similarity_loss = F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0)
#         for b in range(1, B):
#             Similarity_loss += F.cosine_similarity(student_outputs[b, :, :], teacher_outputs[b, :, :], dim=0)
#         Similarity_loss = Similarity_loss / B
#         # Similarity_loss = (F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0) +
#         #                    F.cosine_similarity(
#         #                        student_outputs[1, :, :], teacher_outputs[1, :, :], dim=0)) / 2
#         loss = -torch.mean(Similarity_loss)  # loss = 0 fully same
#         return loss
#
#     def FNKD(self, student_outputs, teacher_outputs, student_feature, teacher_feature):
#         num_classes = student_outputs.shape[1]
#         student_L2norm = torch.norm(student_feature)
#         teacher_L2norm = torch.norm(teacher_feature)
#         if self.sigmoid:
#             q_fn = F.sigmoid(teacher_outputs / teacher_L2norm)
#             to_kd = F.sigmoid(student_outputs / student_L2norm)
#         else:
#             q_fn = F.softmax(teacher_outputs / teacher_L2norm, dim=1)
#             to_kd = F.log_softmax(student_outputs / student_L2norm, dim=1)
#
#         KD_ce_loss = self.ce(
#             to_kd, q_fn)
#         # KD_ce_loss = self.ce(
#         #     q_fn, to_kd[:, 0].long())
#         return KD_ce_loss


class MSKDCAKDLoss(nn.Module):
    def __init__(self):
        super(MSKDCAKDLoss, self).__init__()
        ce_kwargs = {}
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, student_outputs, teacher_outputs):
        loss = 0
        w = [0.4, 0.2, 0.2, 0.2]
        for i in range(0, 4):
            loss += w[i] * (0.1 * self.CAKD(student_outputs[i], teacher_outputs[i])
                            + 0.2 * self.FNKD(student_outputs[i], teacher_outputs[i], student_outputs[i + 4],
                                              teacher_outputs[i + 4]))
        return loss

    def CAKD(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D * W * H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))
        Similarity_loss = (F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0) +
                           F.cosine_similarity(
                               student_outputs[1, :, :], teacher_outputs[1, :, :], dim=0)) / 2
        loss = -torch.mean(Similarity_loss)  # loss = 0 fully same
        return loss

    def FNKD(self, student_outputs, teacher_outputs, student_feature, teacher_feature):
        student_L2norm = torch.norm(student_feature)
        teacher_L2norm = torch.norm(teacher_feature)
        q_fn = F.log_softmax(teacher_outputs / teacher_L2norm, dim=1)
        to_kd = F.softmax(student_outputs / student_L2norm, dim=1)
        KD_ce_loss = self.ce(
            q_fn, to_kd[:, 0].long())
        return KD_ce_loss


class CAKD(nn.Module):
    def __init__(self, interpolate=False, loss_weight=1.0):
        super(CAKD, self).__init__()
        self.interpolate = interpolate
        self.loss_weight = loss_weight

    def forward(self, student_outputs, teacher_outputs):
        if self.interpolate:
            student_outputs = F.interpolate(student_outputs, scale_factor=2, mode='trilinear', align_corners=True)

        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D * W * H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))
        Similarity_loss = F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0)
        for b in range(1, B):
            Similarity_loss += F.cosine_similarity(student_outputs[b, :, :], teacher_outputs[b, :, :], dim=0)
        Similarity_loss = Similarity_loss / B
        # Similarity_loss = (F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0) +
        #                    F.cosine_similarity(
        #                        student_outputs[1, :, :], teacher_outputs[1, :, :], dim=0)) / 2
        loss = torch.mean(Similarity_loss)  # loss = 0 fully same
        return self.loss_weight * loss


class MSKD_CAKD(nn.Module):
    def __init__(self,
                 s_in_chans: int,
                 t_in_chans: int,
                 num_classes: int,
                 tau: float = 1.0,
                 interpolate: bool = False,
                 loss_weight: float = 1.0):
        super(MSKD_CAKD, self).__init__()
        self.num_classes = num_classes
        self.interpolate = interpolate
        self.tau = tau
        self.loss_weight = loss_weight

        self.s_projector = nn.Conv3d(s_in_chans, self.num_classes, 1, 1, 0)
        self.t_projector = nn.Conv3d(t_in_chans, self.num_classes, 1, 1, 0)
        self.s_projector.apply(init_weights)
        self.t_projector.apply(init_weights)

    def forward(self, student_outputs, teacher_outputs):
        if self.interpolate:
            student_outputs = F.interpolate(
                student_outputs, teacher_outputs.shape[-1], mode='trilinear', align_corners=True)

        assert student_outputs.shape[-1] == teacher_outputs.shape[-1], \
            f'shape of student_outputs is {student_outputs.shape[-1]}, ' \
            f'shape of teacher_outputs is {teacher_outputs.shape[-1]}'
        
        student_outputs = self.s_projector(student_outputs)
        teacher_outputs = self.t_projector(teacher_outputs)

        # [B, C, D, W, H] = student_outputs.shape

        # student_outputs = F.softmax(student_outputs, dim=1)
        # student_outputs = student_outputs.reshape(B, C, D * W * H)

        # teacher_outputs = F.softmax(teacher_outputs, dim=1)
        # teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        # with autocast(enabled=False):
        #     student_outputs = torch.bmm(student_outputs, student_outputs.permute(
        #         0, 2, 1))
        #     teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
        #         0, 2, 1))
        # Similarity_loss = F.cosine_similarity(student_outputs, teacher_outputs, dim=0)

        # loss = torch.mean(Similarity_loss)

        C = student_outputs.shape[1]
        softmax_pred_T = F.softmax(teacher_outputs / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        logsoftmax_preds_S = F.log_softmax(student_outputs / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        loss = F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction='mean')
        
        return self.loss_weight * loss


class MSKD_CAKD_V4(nn.Module):
    def __init__(self,
                 s_in_chans: int,
                 t_in_chans: int,
                 num_classes: int,
                 tau: float = 1.0,
                 interpolate: bool = False,
                 loss_weight: float = 1.0):
        super(MSKD_CAKD_V4, self).__init__()
        self.num_classes = num_classes
        self.interpolate = interpolate
        self.tau = tau
        self.loss_weight = loss_weight

        self.s_projector = nn.Conv3d(s_in_chans, self.num_classes, 1, 1, 0)
        self.t_projector = nn.Conv3d(t_in_chans, self.num_classes, 1, 1, 0)
        self.s_projector.apply(init_weights)
        self.t_projector.apply(init_weights)

    def forward(self, student_outputs, teacher_outputs):
        if self.interpolate:
            student_outputs = F.interpolate(
                student_outputs, teacher_outputs.shape[-1], mode='trilinear', align_corners=True)

        assert student_outputs.shape[-1] == teacher_outputs.shape[-1], \
            f'shape of student_outputs is {student_outputs.shape[-1]}, ' \
            f'shape of teacher_outputs is {teacher_outputs.shape[-1]}'
        
        student_outputs = self.s_projector(student_outputs)
        teacher_outputs = self.t_projector(teacher_outputs)

        C = student_outputs.shape[1]
        softmax_pred_T = F.softmax(teacher_outputs / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        logsoftmax_preds_S = F.log_softmax(student_outputs / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        loss = F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction='mean')
        
        return self.loss_weight * loss


class MSKD_CAKD_FNKD(nn.Module):
    def __init__(self,
                 s_in_chans: int,
                 t_in_chans: int,
                 num_classes: int,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 loss_weight: float = 1.0):
        super(MSKD_CAKD_FNKD, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

        self.s_projector = nn.Conv3d(s_in_chans, self.num_classes, 1, 1, 0)
        self.t_projector = nn.Conv3d(t_in_chans, self.num_classes, 1, 1, 0)
        self.s_projector.apply(init_weights)
        self.t_projector.apply(init_weights)
        self.ce = CrossEntropyLoss()

    def forward(self, student_features, teacher_features):
        student_outputs = self.s_projector(student_features)
        teacher_outputs = self.t_projector(teacher_features)

        loss_mskd_cakd = self.alpha * self.mskd_cakd(
            student_outputs, teacher_outputs)
        loss_fnkd = self.beta * self.fnkd(
            student_outputs, teacher_outputs, student_features, teacher_features)

        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/loss_mskd_cakd', loss_mskd_cakd)
        message_hub.update_scalar('train/loss_fnkd', loss_fnkd)

        loss = loss_mskd_cakd + loss_fnkd

        return self.loss_weight * loss

    def mskd_cakd(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D * W * H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))
        Similarity_loss = F.cosine_similarity(student_outputs, teacher_outputs, dim=0)

        loss = torch.mean(Similarity_loss)
        return self.loss_weight * loss

    def fnkd(self, student_outputs, teacher_outputs, student_features, teacher_features):
        student_L2norm = torch.norm(student_features)
        teacher_L2norm = torch.norm(teacher_features)

        q_fn = F.log_softmax(student_outputs / student_L2norm, dim=1)
        to_kd = F.softmax(teacher_outputs / teacher_L2norm, dim=1)

        loss = self.ce(q_fn, to_kd)

        # C = student_outputs.shape[1]
        # softmax_pred_T = F.softmax(teacher_outputs / teacher_L2norm, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # logsoftmax_preds_S = F.log_softmax(student_outputs / student_L2norm, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        #
        # loss = F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction='mean')

        return loss


class MSKD_CAKD_V2(nn.Module):
    def __init__(self,
                 s_in_chans: int,
                 t_in_chans: int,
                 num_classes: int,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 loss_weight: float = 1.0):
        super(MSKD_CAKD_V2, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight

        self.s_projector = nn.Conv3d(s_in_chans, self.num_classes, 1, 1, 0)
        self.t_projector = nn.Conv3d(t_in_chans, self.num_classes, 1, 1, 0)
        self.s_projector.apply(init_weights)
        self.t_projector.apply(init_weights)
        self.ce = CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs):
        student_outputs = self.s_projector(student_outputs)
        teacher_outputs = self.t_projector(teacher_outputs)

        loss_cakd = self.alpha * self.mskd(
            student_outputs, teacher_outputs)
        loss_mskd = self.beta * self.cakd(
            student_outputs, teacher_outputs)

        message_hub = MessageHub.get_current_instance()
        message_hub.update_scalar('train/loss_mskd', loss_mskd)
        message_hub.update_scalar('train/loss_cakd', loss_cakd)

        loss = loss_mskd + loss_cakd

        return self.loss_weight * loss

    def cakd(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D * W * H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D * W * H)

        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))
        Similarity_loss = F.cosine_similarity(student_outputs, teacher_outputs, dim=0)

        loss = torch.mean(Similarity_loss)
        return self.loss_weight * loss

    def mskd(self, student_outputs, teacher_outputs):

        q_fn = F.log_softmax(student_outputs, dim=1)
        to_kd = F.softmax(teacher_outputs, dim=1)

        loss = self.ce(q_fn, to_kd)

        return loss

class FNKD(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FNKD, self).__init__()
        ce_kwargs = {}
        self.ce = CrossEntropyLoss(**ce_kwargs)
        self.loss_weight = loss_weight

    def forward(self, student_outputs, teacher_outputs, student_feature, teacher_feature):
        student_L2norm = torch.norm(student_feature)
        teacher_L2norm = torch.norm(teacher_feature)
        # q_fn = F.log_softmax(teacher_outputs / teacher_L2norm, dim=1)
        # to_kd = F.softmax(student_outputs / student_L2norm, dim=1)

        q_fn = F.log_softmax(student_outputs / student_L2norm, dim=1)
        to_kd = F.softmax(teacher_outputs / teacher_L2norm, dim=1)

        KD_ce_loss = self.ce(q_fn, to_kd)

        # C = student_outputs.shape[1]
        # softmax_pred_T = F.softmax(teacher_outputs / teacher_L2norm, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # logsoftmax_preds_S = F.log_softmax(student_outputs / student_L2norm, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        #
        # loss = F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction='mean')

        return self.loss_weight * KD_ce_loss
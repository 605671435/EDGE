import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from .boundary_loss import BoundaryKDV1
from .hd_loss import LogHausdorffDTLoss


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class TransConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super(TransConv, self).__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose3d(channel_in,
                               channel_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=padding),
            nn.InstanceNorm3d(channel_out),
            nn.PReLU()
        )

    def forward(self, x):
        return self.op(x)

class DSDLoss8(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 kernel_size=3,
                 interpolate=False,
                 bd_include_background=True,
                 hd_include_background=False,
                 one_hot_target=True,
                 sigmoid=False,
                 softmax=True,
                 tau=1,
                 loss_weight: float = 1.0,
                 overall_loss_weight: float = 1.0):
        super(DSDLoss8, self).__init__()
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.tau = tau
        self.bd_include_background = bd_include_background
        self.hd_include_background = hd_include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.overall_loss_weight = overall_loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2

            self.projector = nn.Sequential(
                *up_sample_blks,
            )
        else:
            if self.num_classes == in_chans:
                self.projector = nn.Identity()
            else:
                self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)

        self.projector.apply(init_weights)

        self.bkd = BoundaryKDV1(
            kernel_size=self.kernel_size,
            tau=tau,
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background)
        self.hd = LogHausdorffDTLoss(
            include_background=hd_include_background,
            to_onehot_y=one_hot_target,
            sigmoid=sigmoid,
            softmax=softmax)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        bkd_loss = bkd_loss * self.overall_loss_weight * self.loss_weight
        hd_loss = hd_loss * self.overall_loss_weight * (1 - self.loss_weight)
        return dict(bkd_loss=bkd_loss, hd_loss=hd_loss)

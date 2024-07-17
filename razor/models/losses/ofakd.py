import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from monai.networks import one_hot
from .dsd import TransConv


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


class SepConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 affine=True,
                 instance=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SepTransConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 affine=False,
                 instance=True):
        #   depthwise and pointwise trans-convolution, upsample by 2
        super(SepTransConv, self).__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose3d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_in, affine=affine),
            nn.PReLU(),
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_out, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_out, affine=affine),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.op(x)


class OFALoss(nn.Module):

    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 feature_dim_s: int,
                 feature_dim_t: int,
                 interpolate=False,
                 temperature: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(OFALoss, self).__init__()
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    out_chans = max(feature_dim_s, feature_dim_t)
                else:
                    out_chans = in_chans // 2
                up_sample_blks.append(SepTransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            nn.Conv3d(max(feature_dim_s, feature_dim_t), num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        if self.interpolate:
            feat_student = F.interpolate(feat_student, scale_factor=2, mode='trilinear', align_corners=True)

        logits_student = self.projector(feat_student)

        target_mask = one_hot(label, num_classes=self.num_classes)
        target_mask = target_mask.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
        # b, c, h, w, d = loss.shape
        # loss = loss.reshape(b, c * h * w * d)
        # loss = torch.mean(loss, dim=-1)  # noqa
        return self.loss_weight * loss


class OFALoss8(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 interpolate: bool = False,
                 temperature: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(OFALoss8, self).__init__()
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.eps = eps
        # if self.interpolate:
        #     num_stages += 1

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
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
                # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
            )
        else:
            # self.projector = nn.Identity()
            self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)
            # up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        if self.interpolate:
            # avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, ceil_mode=True)
            # logits_teacher = avg_pool(logits_teacher)
            # label = avg_pool(label)
            # feat_student = self.trans_conv(feat_student)
            feat_student = F.interpolate(feat_student, scale_factor=2, mode='trilinear', align_corners=True)
        logits_student = self.projector(feat_student)

        assert label.shape[-3:] == logits_student.shape[-3:] and \
               logits_student.shape[-3:] == logits_teacher.shape[-3:], \
            f' got shape of label is {label.shape[-3:]}, ' \
            f' shape of logits_student is {logits_student.shape[-3:]},' \
            f' shape of logits_teacher is {logits_teacher.shape[-3:]},' \
            f' them should be equal each other.'

        target_mask = one_hot(label, num_classes=self.num_classes)
        target_mask = target_mask.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
        # b, c, h, w, d = loss.shape
        # loss = loss.reshape(b, c * h * w * d)
        # loss = torch.mean(loss, dim=-1)  # noqa
        return self.loss_weight * loss
# class OFA(BaseDistiller):
#     requires_feat = True
#
#     def __init__(self, student, teacher, criterion, args, **kwargs):
#         super(OFA, self).__init__(student, teacher, criterion, args)
#
#         if len(self.args.ofa_eps) == 1:
#             eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
#             self.args.ofa_eps = eps
#
#         assert len(self.args.ofa_stage) + 1 == len(self.args.ofa_eps)  # +1 for logits
#
#         self.projector = nn.ModuleDict()
#
#         is_cnn_student = is_cnn_model(student)
#
#         _, feature_dim_t = self.teacher.stage_info(-1)
#         _, feature_dim_s = self.student.stage_info(-1)
#
#         for stage in self.args.ofa_stage:
#             _, size_s = self.student.stage_info(stage)
#
#             if is_cnn_student:
#                 in_chans, _, _ = size_s
#
#                 if stage != 4:
#                     down_sample_blk_num = 4 - stage
#                     down_sample_blks = []
#                     for i in range(down_sample_blk_num):
#                         if i == down_sample_blk_num - 1:
#                             out_chans = max(feature_dim_s, feature_dim_t)
#                         else:
#                             out_chans = in_chans * 2
#                         down_sample_blks.append(SepConv(in_chans, out_chans))
#                         in_chans *= 2
#                 else:
#                     down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]
#
#                 projector = nn.Sequential(
#                     *down_sample_blks,
#                     nn.AdaptiveAvgPool2d(1),
#                     nn.Flatten(),
#                     nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes)  # todo: cifar100
#                 )
#             else:
#                 patch_num, embed_dim = size_s
#                 token_num = getattr(student, 'num_tokens', 0)  # cls tokens
#
#                 final_patch_grid = 7  # finally there are 49 patches
#                 patch_grid = int(patch_num ** .5)
#                 merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
#                 merger_modules = []
#                 for i in range(merge_num):
#                     if i == 0:  # proj to feature_dim_s
#                         merger_modules.append(
#                             PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
#                                          dim=embed_dim,
#                                          out_dim=feature_dim_s,
#                                          act_layer=nn.GELU))
#                     else:
#                         merger_modules.append(
#                             PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
#                                          dim=feature_dim_s,
#                                          out_dim=feature_dim_s,
#                                          act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
#                 patch_merger = nn.Sequential(*merger_modules)
#                 blocks = nn.Sequential(
#                     *[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))]  # todo: check this
#                 )
#                 if token_num != 0:
#                     get_feature = nn.Sequential(
#                         TokenFilter(token_num, remove_mode=False),  # todo: token_num > 1
#                         nn.Flatten()
#                     )
#                 else:
#                     get_feature = GAP1d()
#                 projector = nn.Sequential(
#                     TokenFnContext(token_num, patch_merger),
#                     blocks,
#                     get_feature,
#                     nn.Linear(feature_dim_s, args.num_classes)  # todo: cifar100
#                 )
#             set_module_dict(self.projector, stage, projector)
#         self.projector.apply(init_weights)
#         # print(self.projector)  # for debug
#
#     def forward(self, image, label, *args, **kwargs):
#         with torch.no_grad():
#             self.teacher.eval()
#             logits_teacher = self.teacher(image)
#
#         logits_student, feat_student = self.student(image, requires_feat=True)
#
#         num_classes = logits_student.size(-1)
#         if len(label.shape) != 1:  # label smoothing
#             target_mask = F.one_hot(label.argmax(-1), num_classes)
#         else:
#             target_mask = F.one_hot(label, num_classes)
#
#         ofa_losses = []
#         for stage, eps in zip(self.args.ofa_stage, self.args.ofa_eps):
#             idx_s, _ = self.student.stage_info(stage)
#             feat_s = feat_student[idx_s]
#             logits_student_head = get_module_dict(self.projector, stage)(feat_s)
#
#             ofa_losses.append(
#                 ofa_loss(logits_student_head, logits_teacher, target_mask, eps, self.args.ofa_temperature))
#
#         loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)
#
#         loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
#         loss_kd = self.args.kd_loss_weight * ofa_loss(logits_student, logits_teacher, target_mask,
#                                                       self.args.ofa_eps[-1], self.args.ofa_temperature)
#         losses_dict = {
#             "loss_gt": loss_gt,
#             "loss_kd": loss_kd,
#             "loss_ofa": loss_ofa
#         }
#         return logits_student, losses_dict

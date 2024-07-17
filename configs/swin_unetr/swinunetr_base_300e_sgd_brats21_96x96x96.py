from mmengine.config import read_base
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.brats21 import *  # noqa
    from .._base_.schedules.schedule_300e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

dataloader_cfg.update(
    dict(batch_size=2)
)

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=4,
    roi_shapes=roi,
    backbone=dict(
        type=SwinUNETR,
        img_size=roi,
        feature_size=48,
        in_channels=4,
        out_channels=3,
        spatial_dims=3),
    loss_functions=dict(
        type=DiceLoss, to_onehot_y=False, sigmoid=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=2,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

default_hooks.update(
    dict(
        logger=dict(interval=5, val_interval=50),
        checkpoint=dict(type=MyCheckpointHook,
                        by_epoch=True,
                        interval=25,
                        max_keep_ckpts=10,
                        save_best=['Dice'], rule='greater')
    ))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)
from mmengine.config import read_base
from monai.losses import DiceLoss
from seg.models.medical_seg.UXNet_3D.network_backbone import UXNET
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.brats21 import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw_noscheduler import *  # noqa
    from .._base_.monai_runtime import *  # noqa

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=300, val_begin=100, val_interval=25)

dataloader_cfg.update(
    dict(batch_size=2)
)

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=4,
    roi_shapes=roi,
    backbone=dict(
        type=UXNET,
        in_chans=4,
        out_chans=3),
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
    ))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)
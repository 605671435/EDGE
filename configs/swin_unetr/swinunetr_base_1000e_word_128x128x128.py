from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.word import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=1000, val_begin=1000, val_interval=50)


default_hooks.update(dict(
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=True,
                    interval=100,
                    max_keep_ckpts=10,
                    save_best=['Dice'], rule='greater')))

roi = [128, 128, 128]

dataloader_cfg.update(
    dict(num_samples=1),
    # roi size
    roi_x=roi[0],
    roi_y=roi[1],
    roi_z=roi[2],
)

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=17,
    roi_shapes=roi,
    backbone=dict(
        type=SwinUNETR,
        img_size=roi,
        feature_size=48,
        in_channels=1,
        out_channels=17,
        spatial_dims=3),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=1,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))


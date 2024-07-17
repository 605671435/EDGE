from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

dataloader_cfg.update(
    dict(num_samples=2)
)

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=3000, val_begin=200, val_interval=100)

param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(type=LinearLR,
         start_factor=1e-6,
         by_epoch=True,
         begin=0,
         end=50),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type=CosineAnnealingLR,
         T_max=2950,
         by_epoch=True,
         begin=50,
         end=3000)]

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=SwinUNETR,
        img_size=roi,
        feature_size=12,
        in_channels=1,
        out_channels=14,
        spatial_dims=3),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=2,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))


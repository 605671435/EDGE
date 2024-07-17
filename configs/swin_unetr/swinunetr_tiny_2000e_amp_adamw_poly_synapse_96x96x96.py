from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiSeg
from torch.optim import AdamW
from mmengine.optim import AmpOptimWrapper

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_2000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=1e-4, weight_decay=1e-5))

dataloader_cfg.update(
    dict(num_samples=2)
)

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

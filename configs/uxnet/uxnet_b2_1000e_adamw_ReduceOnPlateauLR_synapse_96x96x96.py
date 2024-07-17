from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.medical_seg.UXNet_3D.network_backbone import UXNET
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw_ReduceOnPlateauLR import *  # noqa
    from .._base_.monai_runtime import *  # noqa

dataloader_cfg.update(
    dict(num_samples=2)
)

val_evaluator.update(
    dict(
        metrics=dict(
            metrics=['Dice'])))

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=UXNET,
        in_chans=1,
        out_chans=14,
    ),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=2,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

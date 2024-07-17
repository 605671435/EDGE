from torch.optim import AdamW
from mmengine.optim.scheduler import PolyLR
from mmengine.optim import AmpOptimWrapper
from mmengine.hooks import IterTimerHook, ParamSchedulerHook, DistSamplerSeedHook, EmptyCacheHook
from mmseg.engine.hooks import SegVisualizationHook
from seg.engine.hooks import MyCheckpointHook
from seg.engine.hooks.logger_hook import MyLoggerHook
from mmengine.runner.loops import EpochBasedTrainLoop
from seg.engine.runner.monai_loops import MonaiValLoop, MonaiTestLoop

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=3000, val_begin=200, val_interval=100)
val_cfg = dict(type=MonaiValLoop)
test_cfg = dict(type=MonaiTestLoop)
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=1e-4, weight_decay=1e-5))

# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        # eta_min=1e-4,
        power=0.9,
        begin=0,
        end=3000,
        by_epoch=True)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=MyLoggerHook, interval=10, val_interval=1),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=True,
                    interval=10,
                    max_keep_ckpts=1,
                    save_best=['Dice'], rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

custom_hooks = [
    dict(
        type=EmptyCacheHook,
        before_epoch=False,
        after_epoch=True,
        after_iter=False)]

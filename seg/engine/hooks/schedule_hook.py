# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from seg.registry import HOOKS
from mmengine.logging import MMLogger, print_log, MessageHub


@HOOKS.register_module()
class TrainingScheduleHook(Hook):
    def __init__(self,
                 interval,
                 use_fcn=False):
        self.interval = interval
        self.use_fcn = use_fcn

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

        if runner.iter == self.interval:
            new_plugins = [dict(cfg=dict(type='EX_Module', with_self=True),
                                                  stages=(True, True, True, True),
                                                  position='after_conv1')]
            runner.model.backbone.plugins = new_plugins

            for i, layer_name in enumerate(runner.model.backbone.res_layers):
                res_layer = getattr(runner.model.backbone, layer_name)
                for block in res_layer:
                    if hasattr(block, "ex_module") and block.ex_module.use_self is True:
                        block.ex_module.with_self = True
            logger: MMLogger = MMLogger.get_current_instance()
            if self.use_fcn:
                runner.model.decode_head.with_self = True
                print_log(f'decode_head.with_self change to {True}', logger)
            print_log(f'plugins change to {new_plugins}', logger)


class LossWeightScheduleHook(Hook):

    def __init__(self,
                 eta_min=0.001,
                 gamma=0.001):

        self.eta_min = eta_min
        self.gamma = gamma

    def after_train_epoch(self, runner) -> None:
        loss_weights = getattr(runner.model, 'loss_weights')
        message_hub = MessageHub.get_current_instance()
        alpha = loss_weights[0]
        # alpha -= 0.001
        # if alpha <= 0.001:
        #     alpha = 0.001
        alpha -= self.gamma
        if alpha <= self.eta_min:
            alpha = self.eta_min
        loss_weights[0] = alpha
        loss_weights[1] = 1 - alpha
        message_hub.update_scalar('train/dice_weight', loss_weights[0])
        message_hub.update_scalar('train/hd_weight', loss_weights[1])


class SingleLossWeightScheduleHook2(Hook):

    def __init__(self,
                 eta_min=0.001,
                 gamma=0.001):

        self.eta_min = eta_min
        self.gamma = gamma

    def after_train_epoch(self, runner) -> None:
        loss_weights = getattr(runner.model, 'loss_weights')
        message_hub = MessageHub.get_current_instance()
        alpha = loss_weights[1]
        # alpha -= 0.001
        # if alpha <= 0.001:
        #     alpha = 0.001
        alpha += self.gamma
        if alpha > self.eta_min:
            alpha = self.eta_min
        # loss_weights[0] = alpha
        loss_weights[1] = alpha
        # message_hub.update_scalar('train/dice_weight', loss_weights[0])
        message_hub.update_scalar('train/hd_weight', loss_weights[1])


class DistillLossWeightScheduleHook(Hook):

    def __init__(self,
                 eta_min=0.001,
                 alpha=1.0,
                 gamma=0.001):

        self.eta_min = eta_min
        self.alpha = alpha
        self.gamma = gamma
        self.set = False

    def before_train(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        distill_losses['loss_boundary'].loss_weight = self.alpha
        distill_losses['loss_hd'].loss_weight = 1 - self.alpha
        message_hub.update_scalar('train/dice_weight', self.alpha)
        message_hub.update_scalar('train/hd_weight', 1 - self.alpha)

    def after_train_epoch(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        self.alpha -= self.gamma
        if self.alpha <= self.eta_min:
            self.alpha = self.eta_min
        distill_losses['loss_boundary'].loss_weight = self.alpha
        distill_losses['loss_hd'].loss_weight = 1 - self.alpha
        message_hub.update_scalar('train/dice_weight', self.alpha)
        message_hub.update_scalar('train/hd_weight', 1 - self.alpha)


class DistillLossWeightScheduleHookV2(Hook):

    def __init__(self,
                 loss_names,
                 eta_min=0.001,
                 alpha=1.0,
                 gamma=0.001):

        self.loss_names = loss_names if isinstance(loss_names, list) else [loss_names]
        self.eta_min = eta_min
        self.alpha = alpha
        self.gamma = gamma
        self.set = False

    def before_train(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        for loss_name in self.loss_names:
            if f'train/{loss_name}_weight' in message_hub.log_scalars:
                self.alpha = message_hub.get_scalar(f'train/{loss_name}_weight').current()
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.info(f'resumed loss weight: train/{loss_name}_weight, value: {self.alpha}')
            else:
                message_hub.update_scalar(f'train/{loss_name}_weight', self.alpha)
            distill_losses[loss_name].loss_weight = self.alpha

    def after_train_epoch(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        self.alpha -= self.gamma
        if self.alpha <= self.eta_min:
            self.alpha = self.eta_min
        for loss_name in self.loss_names:
            distill_losses[loss_name].loss_weight = self.alpha
            message_hub.update_scalar(f'train/{loss_name}_weight', self.alpha)


class SingleLossWeightScheduleHook(Hook):

    def __init__(self,
                 eta_min=0.001,
                 alpha=1.0,
                 gamma=0.001):

        self.eta_min = eta_min
        self.alpha = alpha
        self.gamma = gamma
        self.set = False

    def before_train(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        distill_losses['loss_hd'].loss_weight = 1 - self.alpha
        message_hub.update_scalar('train/hd_weight', 1 - self.alpha)

    def after_train_epoch(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        self.alpha -= self.gamma
        if self.alpha <= self.eta_min:
            self.alpha = self.eta_min
        distill_losses['loss_hd'].loss_weight = 1 - self.alpha
        message_hub.update_scalar('train/hd_weight', 1 - self.alpha)


class SwitchLossScheduleHook(Hook):

    def __init__(self,
                 stop_epoch=500,
                 loss_name='loss_ofa'):

        self.stop_epoch = stop_epoch
        self.set = False
        self.loss_name = loss_name

    def after_train_epoch(self, runner) -> None:
        message_hub = MessageHub.get_current_instance()
        cur_epoch = message_hub.get_info('epoch')
        if cur_epoch >= self.stop_epoch and not self.set:
            distill_losses = getattr(runner.model.distiller, 'distill_losses')
            distill_losses[self.loss_name].loss_weight = 0
            self.set = True

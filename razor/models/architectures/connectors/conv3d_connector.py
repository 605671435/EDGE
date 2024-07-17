# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrazor.models.architectures.connectors.base_connector import BaseConnector


class Conv3DConnector(BaseConnector):

    def __init__(self,
                 args: Dict = {},
                 interpolate_size=None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        self.func = nn.Conv3d(**args)
        self.interpolate_size = interpolate_size

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input features.
        """
        x = self.func(x)
        if self.interpolate_size is not None:
            x = F.interpolate(x, self.interpolate_size, mode='trilinear')
        return x

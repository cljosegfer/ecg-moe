
import torch

from dataclasses import dataclass
from typing import Any
from configs.fake import Downstream_cnn_args

@dataclass
class MoE_cnn_args: 
    gate_path: str = 'output/gate.pt'
    resnet_config: Any = Downstream_cnn_args()
    n_experts: int = 3
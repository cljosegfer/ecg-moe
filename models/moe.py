
import torch
import torch.nn as nn

from models.baseline import ResnetBaseline

class ResnetMoE(nn.Module):
    def __init__(self, gate_path, resnet_config, n_experts):
        super().__init__()

        self.gate = torch.load(gate_path)
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            self.experts.append(ResnetBaseline(**resnet_config.__dict__))
        self.num_classes = resnet_config.__dict__['n_classes']


    def forward(self, x):
        g = self.gate.forward(x)
        logits = [expert.forward(x) for expert in self.experts]

        g = g.unsqueeze(1)
        g = g.expand(-1, self.num_classes, -1)
        logits = torch.stack(logits, dim = 2)
        logits = torch.sum(g * logits, dim = 2)

        return logits
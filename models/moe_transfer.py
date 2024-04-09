
import torch
import torch.nn as nn

from models.baseline import ResnetBaseline
from collections import OrderedDict

class ResnetMoE(nn.Module):
    def __init__(self, gate_path, resnet_config, n_experts):
        super().__init__()

        self.gate = torch.load(gate_path)
        backbone = self.generate_backbone(resnet_config)
        # self.gate = ResnetBaseline(**resnet_config.__dict__)
        # n_experts = 6
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            # self.experts.append(ResnetBaseline(**resnet_config.__dict__))
            expert = ResnetBaseline(**resnet_config.__dict__)
            log = expert.load_state_dict(backbone, strict = False)
            assert log.missing_keys == ['linear.weight', 'linear.bias']
            self.experts.append(expert)
        self.num_classes = resnet_config.__dict__['n_classes']


    def forward(self, x):
        g = self.gate.forward(x)
        g = torch.sigmoid(g)
        logits = [expert.forward(x) for expert in self.experts]

        g = g.unsqueeze(1)
        g = g.expand(-1, self.num_classes, -1)
        logits = torch.stack(logits, dim = 2)
        logits = torch.sum(g * logits, dim = 2)

        return logits
    
    def generate_backbone(self, resnet_config):
        key_transformation = []
        for key in ResnetBaseline(**resnet_config.__dict__).state_dict().keys():
            key_transformation.append(key)
        
        backbone = nn.Sequential(*list(self.gate.children())[:-1])

        state_dict = backbone.state_dict()
        new_state_dict = OrderedDict()

        for i, (key, value) in enumerate(state_dict.items()):
            new_key = key_transformation[i]
            new_state_dict[new_key] = value
        
        return new_state_dict

import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,}trainable parameters')
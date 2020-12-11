import torch

class Wrap(torch.nn.Module):
    def __init__(self, m, eg):
        super().__init__()
        self.m = m
        self.eg = eg
    def forward(self):
        return [r.bbox for r in self.m(*self.eg)]
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_size: int = 512, num_classes: int = 100,
                 margin: float = 0.5, scale: float = 64):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings_norm, weight_norm)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cos_theta)
        return output * self.scale


class FullModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None):
        emb = self.backbone(x)
        if labels is not None:
            return self.head(emb, labels)
        return emb

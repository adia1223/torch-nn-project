import torch
from torch import nn

from models.images.classification.few_shot_learning import DistanceBasedFSLSolution


class ProtoNet(DistanceBasedFSLSolution):
    def __init__(self, backbone: nn.Module):
        super(ProtoNet, self).__init__()
        self.feature_extractor = backbone

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(batch)

    def get_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        return torch.mean(support_set, dim=1)

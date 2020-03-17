import torch
from torch import nn


class FewShotLearningSolution(nn.Module):
    def __init__(self):
        super(FewShotLearningSolution, self).__init__()

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DistanceBasedFSLSolution(FewShotLearningSolution):
    def __init__(self):
        super(DistanceBasedFSLSolution, self).__init__()

        self.n_classes = None
        self.support_set_size = None
        self.support_set_features = None
        self.class_prototypes = None
        self.query_set_features = None
        self.query_set_size = None

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_prototypes(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # TODO
    # def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     raise NotImplementedError

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        self.n_classes = support_set.size(0)
        self.support_set_size = support_set.size(1)
        self.query_set_size = query_set.size(0)

        self.support_set_features = self.extract_features(support_set.squeeze(0)).view(self.n_classes,
                                                                                       self.support_set_size, -1)

        self.query_set_features = self.extract_features(query_set)

        self.class_prototypes = self.get_prototypes(self.support_set_features, self.query_set_features)

        query_set_features_prepared = self.query_set_features.unsqueeze(1).repeat_interleave(repeats=self.n_classes,
                                                                                             dim=1)

        distance = torch.sum((self.classes_features.unsqueeze(0).repeat_interleave(repeats=self.query_set_size,
                                                                                   dim=0) -
                              query_set_features_prepared).pow(2), dim=2)

        return distance

import os

import torch
import torch.nn.functional as F

from models.images.classification.few_shot_learning import FitTransformFewShotLearningSolution
from sessions import Session


class RandomClassifier(FitTransformFewShotLearningSolution):
    def __init__(self):
        super().__init__()
        self.n_classes = 1
        self.n_query = 1

    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor) -> torch.Tensor:
        self.n_classes = support_set.size(0)
        self.n_query = query_set.size(0)

        return F.softmax(torch.rand(self.n_query, self.n_classes), dim=1)

    def fit(self, support_set: torch.Tensor):
        self.n_classes = support_set.size(0)

    def transform(self, x: torch.Tensor):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)
        self.n_query = x.size(0)
        prob = F.softmax(torch.rand(self.n_query, self.n_classes), dim=1)
        if prob.size(0) == 1:
            prob = torch.squeeze(prob, 0)
        return prob


if __name__ == '__main__':
    best_model = RandomClassifier()

    session_info = {
        "task": "few-shot learning",
        "model": "RANDOM",
    }

    session = Session()
    session.build(name="RANDOM", comment=r"RANDOM classifier",
                  **session_info)

    torch.save(best_model, os.path.join(session.data['output_dir'], "trained_model_state_dict.tar"))
    session.save_info()

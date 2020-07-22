# noinspection PyUnresolvedReferences
import json
import os

import torch

from inference.inference_config import NAME2FOLDER
from models.images.classification.few_shot_learning import FitTransformFewShotLearningSolution


def get_model(model_name):
    if model_name not in NAME2FOLDER:
        return None
    model_folder = NAME2FOLDER[model_name]
    model_file = os.path.join(model_folder, 'output', 'trained_model_state_dict.tar')
    model = torch.load(model_file)
    model.eval()

    info_file = os.path.join(model_folder, 'output', 'info.json')
    with open(info_file) as fin:
        info = json.load(fin)

    return model, info


def fit_model(model_name: str, task: torch.Tensor):
    model, _ = get_model(model_name)
    if model is None or not isinstance(model, FitTransformFewShotLearningSolution):
        raise NotImplementedError('Model "%s" not found or can not be applied' % model_name)
    model.fit(task)
    return model


def apply_model(model: FitTransformFewShotLearningSolution, query: torch.Tensor):
    result = model.transform(query)

    return result

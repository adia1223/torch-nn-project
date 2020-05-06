import json

# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.mctdfmn import *
# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.protonet import *
# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.triplet import *


def change_dataset(model_folder: str, dataset_name: str, record: int, val_batch_size: int = None,
                   val_n_way: int = None, balanced_batches: bool = None):
    model_file = os.path.join(model_folder, 'output', 'trained_model_state_dict.tar')
    info_file = os.path.join(model_folder, 'output', 'info.json')
    with open(info_file) as fin:
        info = json.load(fin)
    print(info)
    if val_batch_size is not None:
        info['val_batch_size'] = val_batch_size
    if val_n_way is not None:
        info['n_way'] = val_n_way
    if balanced_batches is not None:
        info['balanced_batches'] = balanced_batches
    model = torch.load(model_file)
    model.eval()

    dataset = LABELED_DATASETS[dataset_name](augment_prob=0, image_size=info['image_size']).subdataset
    sampler = FSLEpisodeSampler(subdataset=dataset, n_way=info['n_way'], n_shot=info['n_shot'],
                                batch_size=info['val_batch_size'],
                                balanced=info['balanced_batches'])
    info['dataset'] += '->' + dataset_name
    info['record'] = record
    print(info['dataset'])

    score = evaluate_solution_episodes(model, sampler)
    info['accuracy'] = score

    # info.pop('name')
    info.pop('comment')
    session = Session()
    session.build(name=info['screen_name'] + '_transfer',
                  comment=r"Few-Shot Learning solution from '" + info['full_name'] + "'",
                  **info)
    torch.save(model, os.path.join(session.data['output_dir'], "trained_model_state_dict.tar"))
    session.save_info()
    # print(evaluate_solution(model, val_sampler, n_iterations=600))


if __name__ == '__main__':
    # paths = [r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_637276-53-47-18-06-04-2020',
    #          # 1-shot miniImageNet
    #          r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_055670-43-55-20-05-04-2020',
    #          # 5-shot miniImageNet
    #          ]

    # paths = [r'D:\petrtsv\projects\ds\pytorch-sessions\ProtoNet\ProtoNet_661818-03-58-00-10-04-2020',
    #          # 1-shot miniImageNet
    #          r'D:\petrtsv\projects\ds\pytorch-sessions\ProtoNet\ProtoNet_312646-55-13-05-10-04-2020'
    #          # 5-shot miniImageNet
    #          ]

    # paths = [r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_424847-30-38-05-01-05-2020',
    #          # 1-shot google-landmarks 15-way no ts
    #          r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_674842-01-41-11-01-05-2020'
    #          # 5-shot google-landmarks 15-way no ts
    #          ]

    paths = [r'D:\petrtsv\projects\ds\pytorch-sessions\TripletNet\TripletNet_214392-04-52-05-04-05-2020',
             # 1-shot google-landmarks
             # r''
             # # ...
             ]

    DATASET_NAME = 'google-landmarks'
    RECORD = 530

    for path in paths:
        print(path)
        change_dataset(path, DATASET_NAME, RECORD, val_batch_size=5, val_n_way=900, balanced_batches=False)
        print()

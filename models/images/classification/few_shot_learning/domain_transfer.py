import json

# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.dummy import *
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

    if 'image_size' not in info:
        info['image_size'] = 0

    if 'n_shot' not in info:
        info['n_shot'] = 1

    if 'dataset' not in info:
        info['dataset'] = 'none'

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

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_704733-26-18-15-22-05-2020',
    #     # 1-shot google-landmarks 15-way no ts
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_396976-51-31-06-23-05-2020'
    #     # 5-shot google-landmarks 15-way no ts
    # ]

    # paths = [r'D:\petrtsv\projects\ds\pytorch-sessions\TripletNet\TripletNet_214392-04-52-05-04-05-2020',
    #          # 1-shot google-landmarks
    #          # r''
    #          # # ...
    #          ]

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_810889-42-11-03-22-05-2020',
    #     # 5-shot google-landmarks + google-landmarks-selfsupervised 15-way no ts
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_375041-53-19-07-22-05-2020'
    #     # 1-shot google-landmarks + google-landmarks-selfsupervised 15-way no ts
    # ]

    # paths = [
    #     # r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_826860-08-48-08-27-05-2020',
    #     # 1-shot google-landmarks-selfsupervised->google-landmarks 15-way no ts R750
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_565503-44-34-07-28-05-2020'
    #     # 5-shot google-landmarks-selfsupervised->google-landmarks 15-way no ts R760
    # ]

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_216898-02-08-07-16-06-2020',
    #     # 1-shot google-landmarks 15-way no ts no scaling
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_606202-24-36-11-16-06-2020'
    #     # 5-shot google-landmarks 15-way no ts no scaling
    # ]

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_959766-57-51-01-17-06-2020',
    #     # 1-shot google-landmarks 15-way extended input
    #     # r''
    #     # 5-shot google-landmarks 15-way extended input
    # ]

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_606743-06-04-06-19-06-2020',
    #     # 1-shot google-landmarks 15-way extended input PCA
    #     # r''
    #     # 5-shot google-landmarks 15-way extended input PCA
    # ]

    # paths = [
    #     r'D:\petrtsv\projects\ds\pytorch-sessions\RANDOM\RANDOM_203154-58-22-18-18-07-2020',
    #     # RANDOM
    # ]

    paths = [
        r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_742585-10-48-10-23-10-2020'
    ]

    DATASET_NAME = 'miniImageNet-test'
    RECORD = -1000

    for path in paths:
        print(path)
        change_dataset(path, DATASET_NAME, RECORD, val_batch_size=5, val_n_way=5, balanced_batches=True)
        print()

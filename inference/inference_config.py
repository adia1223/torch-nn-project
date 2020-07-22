# import models' classes

# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.dummy import *
# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.mctdfmn import *
# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.protonet import *
# noinspection PyUnresolvedReferences
from models.images.classification.few_shot_learning.triplet import *

# Add new models here
NAME2FOLDER = {
    'dfmn-landmarks-1shot':
        r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_826860-08-48-08-27-05-2020',
    'dfmn-landmarks-5shot':
        r'D:\petrtsv\projects\ds\pytorch-sessions\FSL_MCTDFMN\FSL_MCTDFMN_565503-44-34-07-28-05-2020',
    'random':
        r'D:\petrtsv\projects\ds\pytorch-sessions\RANDOM\RANDOM_203154-58-22-18-18-07-2020',
}

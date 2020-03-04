import time

from sessions.transfer_learning_classification import GoogLeNetGTSRB, ResNet18GTSRB, ResNet18CIFAR10, \
    ResNet18CosineGTSRB, ResNet18CosineCIFAR10

EPOCHS = 25
LR = 0.001
BATCH_SIZE = 16

RECORD = 0
SOFTMAX = 0

FREEZE_BACKBONE = False
PRETRAINED = True


# SHARES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98)
#
# SAMPLES_EPOCHS = 10
# SAMPLES = (
#     (50, SAMPLES_EPOCHS, 1),
#     (25, SAMPLES_EPOCHS * 2, 2),
#     (10, SAMPLES_EPOCHS * 5, 5),
#     # (5, SAMPLES_EPOCHS * 10, 10),
#     # (1, SAMPLES_EPOCHS * 50, 50)
# )


def fill_info(session, backbone, dataset, reduce, frozen, lr, epochs, batch_size, aug_prob, start_time, softmax,
              record, pretrained, cosine):
    session.info['backbone'] = backbone
    session.info['dataset'] = dataset
    session.info['reduced'] = reduce
    session.info['frozen_feature_extractor'] = frozen
    session.info['lr'] = lr
    session.info['epochs'] = epochs
    session.info['batch_size'] = batch_size
    session.info['augment_prob'] = aug_prob
    session.info['start_time'] = start_time
    session.info['softmax'] = softmax
    session.info['record'] = record
    session.info['pretrained'] = pretrained
    session.info['cosine'] = cosine


def googlenet_gtsrb(aug_prob=0.0, freeze_backbone=FREEZE_BACKBONE, reduce=0.0, eval_period=1, batch_size=BATCH_SIZE,
                    lr=LR,
                    epochs=EPOCHS,
                    softmax=SOFTMAX,
                    pretrained=PRETRAINED,
                    ):
    print("GoogLeNet on GTSRB dataset (reduce = %.2f)" % reduce)
    dataset_size = str(int((1 - reduce) * 100)) + "%" if reduce < 1 else str(int(reduce))
    session = GoogLeNetGTSRB(name="GoogLeNet_%s_GTSRB" % dataset_size, epochs=epochs, learning_rate=lr,
                             comment="GoogLeNet on GTSRB dataset (reduce = %.2f)" % reduce,
                             batch_size=batch_size, freeze_backbone=freeze_backbone,
                             reduce=reduce, eval_period=eval_period, augment_prob=aug_prob, dataloader_workers=4,
                             val_batch_size=BATCH_SIZE, pretrained=pretrained)
    session.dataset.label_stat()
    fill_info(
        session,
        'GoogLeNet',
        'GTSRB',
        reduce,
        int(freeze_backbone),
        lr,
        epochs,
        batch_size,
        aug_prob,
        time.ctime(),
        softmax,
        RECORD,
        pretrained,
        0,
    )
    session.run()
    session.save_info()


def resnet18_gtsrb(aug_prob=0.0, freeze_backbone=FREEZE_BACKBONE, reduce=0.0, eval_period=1, batch_size=BATCH_SIZE,
                   lr=LR,
                   epochs=EPOCHS,
                   softmax=SOFTMAX,
                   pretrained=PRETRAINED,
                   ):
    print("ResNet18 on GTSRB dataset (reduce = %.2f)" % reduce)
    dataset_size = str(int((1 - reduce) * 100)) + "%" if reduce < 1 else str(int(reduce))
    session = ResNet18GTSRB(name="ResNet18_%s_GTSRB" % dataset_size, epochs=epochs, learning_rate=lr,
                            comment="ResNet18 on GTSRB dataset (reduce = %.2f)" % reduce,
                            batch_size=batch_size, freeze_backbone=freeze_backbone,
                            reduce=reduce, eval_period=eval_period, augment_prob=aug_prob, dataloader_workers=4,
                            val_batch_size=BATCH_SIZE, pretrained=pretrained)
    session.dataset.label_stat()
    fill_info(
        session,
        'ResNet18',
        'GTSRB',
        reduce,
        int(freeze_backbone),
        lr,
        epochs,
        batch_size,
        aug_prob,
        time.ctime(),
        softmax,
        RECORD,
        pretrained,
        0,
    )
    session.run()
    session.save_info()


def resnet18_cifar10(aug_prob=0.0, freeze_backbone=FREEZE_BACKBONE, reduce=0.0, eval_period=1, batch_size=BATCH_SIZE,
                     lr=LR,
                     epochs=EPOCHS,
                     softmax=SOFTMAX,
                     pretrained=PRETRAINED,
                     ):
    print("ResNet18 on CIFAR10 dataset (reduce = %.2f)" % reduce)
    dataset_size = str(int((1 - reduce) * 100)) + "%" if reduce < 1 else str(int(reduce))
    session = ResNet18CIFAR10(name="ResNet18_%s_CIFAR10" % dataset_size, epochs=epochs, learning_rate=lr,
                              comment="ResNet18 on CIFAR10 dataset (reduce = %.2f)" % reduce,
                              batch_size=batch_size, freeze_backbone=freeze_backbone,
                              reduce=reduce, eval_period=eval_period, dataloader_workers=4,
                              augment_prob=aug_prob,
                              val_batch_size=BATCH_SIZE, pretrained=pretrained)
    session.dataset.label_stat()
    fill_info(
        session,
        'ResNet18',
        'CIFAR10',
        reduce,
        int(freeze_backbone),
        lr,
        epochs,
        batch_size,
        aug_prob,
        time.ctime(),
        softmax,
        RECORD,
        pretrained=pretrained,
        cosine=0,
    )
    session.run()
    session.save_info()


def resnet18cosine_gtsrb(aug_prob=0.0, freeze_backbone=FREEZE_BACKBONE, reduce=0.0, eval_period=1,
                         batch_size=BATCH_SIZE,
                         lr=LR,
                         epochs=EPOCHS,
                         softmax=SOFTMAX,
                         pretrained=PRETRAINED,
                         ):
    print("ResNet18Cosine on GTSRB dataset (reduce = %.2f)" % reduce)
    dataset_size = str(int((1 - reduce) * 100)) + "%" if reduce < 1 else str(int(reduce))
    session = ResNet18CosineGTSRB(name="ResNet18Cosine_%s_GTSRB" % dataset_size, epochs=epochs, learning_rate=lr,
                                  comment="ResNet18Cosine on GTSRB dataset (reduce = %.2f)" % reduce,
                                  batch_size=batch_size, freeze_backbone=freeze_backbone,
                                  reduce=reduce, eval_period=eval_period, augment_prob=aug_prob, dataloader_workers=4,
                                  val_batch_size=BATCH_SIZE, pretrained=pretrained)
    session.dataset.label_stat()
    fill_info(
        session,
        'ResNet18Cosine',
        'GTSRB',
        reduce,
        int(freeze_backbone),
        lr,
        epochs,
        batch_size,
        aug_prob,
        time.ctime(),
        softmax,
        RECORD,
        pretrained,
        1,
    )
    session.run()
    session.save_info()


def resnet18cosine_cifar10(aug_prob=0.0, freeze_backbone=FREEZE_BACKBONE, reduce=0.0, eval_period=1,
                           batch_size=BATCH_SIZE,
                           lr=LR,
                           epochs=EPOCHS,
                           softmax=SOFTMAX,
                           pretrained=PRETRAINED,
                           ):
    print("ResNet18Cosine on CIFAR10 dataset (reduce = %.2f)" % reduce)
    dataset_size = str(int((1 - reduce) * 100)) + "%" if reduce < 1 else str(int(reduce))
    session = ResNet18CosineCIFAR10(name="ResNet18Cosine_%s_CIFAR10" % dataset_size, epochs=epochs, learning_rate=lr,
                                    comment="ResNet18Cosine on CIFAR10 dataset (reduce = %.2f)" % reduce,
                                    batch_size=batch_size, freeze_backbone=freeze_backbone,
                                    reduce=reduce, eval_period=eval_period, dataloader_workers=4,
                                    augment_prob=aug_prob,
                                    val_batch_size=BATCH_SIZE, pretrained=pretrained)
    session.dataset.label_stat()
    fill_info(
        session,
        'ResNet18Cosine',
        'CIFAR10',
        reduce,
        int(freeze_backbone),
        lr,
        epochs,
        batch_size,
        aug_prob,
        time.ctime(),
        softmax,
        RECORD,
        pretrained=pretrained,
        cosine=1,
    )
    session.run()
    session.save_info()


if __name__ == '__main__':
    AUG = 0.5
    EPOCHS_NUM_MN = 1200
    EPOCHS_NUM_MX = 2000

    # resnet18cosine_cifar10(reduce=1, epochs=EPOCHS_NUM_MX, freeze_backbone=False, eval_period=10, batch_size=4,
    #                        aug_prob=AUG)
    # resnet18cosine_cifar10(reduce=1, epochs=EPOCHS_NUM_MX, freeze_backbone=True, eval_period=10, batch_size=4,
    #                        aug_prob=AUG)
    #
    # resnet18cosine_cifar10(reduce=5, epochs=EPOCHS_NUM_MN, freeze_backbone=False, eval_period=10, batch_size=4,
    #                        aug_prob=AUG)
    # resnet18cosine_cifar10(reduce=5, epochs=EPOCHS_NUM_MN, freeze_backbone=True, eval_period=10, batch_size=4,
    #                        aug_prob=AUG)
    #
    # resnet18cosine_gtsrb(reduce=1, epochs=EPOCHS_NUM_MX, freeze_backbone=False, eval_period=10, batch_size=4, aug_prob=AUG)
    resnet18cosine_gtsrb(reduce=1, epochs=EPOCHS_NUM_MX, freeze_backbone=True, eval_period=10, batch_size=4,
                         aug_prob=AUG)

    resnet18cosine_gtsrb(reduce=5, epochs=EPOCHS_NUM_MN, freeze_backbone=False, eval_period=10, batch_size=4,
                         aug_prob=AUG)
    resnet18cosine_gtsrb(reduce=5, epochs=EPOCHS_NUM_MN, freeze_backbone=True, eval_period=10, batch_size=4,
                         aug_prob=AUG)

    # resnet18_gtsrb(reduce=1, epochs=400, freeze_backbone=False, eval_period=10, batch_size=4, aug_prob=AUG)
    # resnet18_gtsrb(reduce=1, epochs=400, freeze_backbone=True, eval_period=10, batch_size=4, aug_prob=AUG)
    #
    # resnet18_gtsrb(reduce=5, epochs=400, freeze_backbone=False, eval_period=10, batch_size=4, aug_prob=AUG)
    # resnet18_gtsrb(reduce=5, epochs=400, freeze_backbone=True, eval_period=10, batch_size=4, aug_prob=AUG)

    # resnet18_cifar10(reduce=1, epochs=400, freeze_backbone=False, eval_period=10, batch_size=4, aug_prob=AUG)
    # resnet18_cifar10(reduce=1, epochs=400, freeze_backbone=True, eval_period=10, batch_size=4, aug_prob=AUG)
    #
    # resnet18_cifar10(reduce=5, epochs=400, freeze_backbone=False, eval_period=10, batch_size=4, aug_prob=AUG)
    # resnet18_cifar10(reduce=5, epochs=400, freeze_backbone=True, eval_period=10, batch_size=4, aug_prob=AUG)
    # for reduce, epochs, eval_period in zip(
    #         (
    #                 0.0, 0.5, 0.9, 0.95, 500, 100,
    #                 50,
    #                 10,
    #         ),
    #         (
    #                 10, 20, 30, 40, 50, 60,
    #                 70,
    #                 80,
    #         ),
    #         (
    #                 1, 2, 3, 4, 5, 6,
    #                 7,
    #                 8,
    #         )):
    #
    #     for freeze in (True, False):
    #         resnet18_cifar10(reduce=reduce, epochs=epochs, freeze_backbone=freeze, eval_period=eval_period,
    #                          aug_prob=AUG)
    #
    # for reduce, epochs, eval_period in zip(
    #         (
    #                 0.0, 0.5, 0.9, 0.95, 200, 100,
    #                 50,
    #                 10,
    #         ),
    #         (
    #                 15, 30, 45, 60, 75, 90,
    #                 105,
    #                 120,
    #         ),
    #         (
    #                 1, 2, 3, 4, 5, 6,
    #                 7,
    #                 8,
    #         )):
    #     resnet18_gtsrb(reduce=reduce, epochs=epochs, freeze_backbone=False, eval_period=eval_period, aug_prob=AUG)
    #     resnet18_gtsrb(reduce=reduce, epochs=epochs * 2, freeze_backbone=True, eval_period=eval_period * 2,
    #                    aug_prob=AUG)

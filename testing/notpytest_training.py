# import os
# import random
#
# import torch
# from torch import optim, nn
# from torch.utils.data import DataLoader
#
# from data.celeba import CelebaCroppedDataset
# from data.image_transforms import scaling_grayscale
# from models.images.generative.gan.dcgan import DCGANGenerator, DCGANDiscriminator
# from models.images.generative.gan.resgan import ResGenerator, ResDiscriminator
# from training import adversarial
#
# MANUAL_SEED = 666
#
# nz_size = 128
#
# fake_label = 0
# real_label = 1
#
# os.environ['TORCH_HOME'] = "D:\\torch_home"
# # CHECKPOINTS_PATH = "D:\\petrtsv\\projects\\ds\\SkyGAN\\checkpoints"
# # TRAINED_PATH = "D:\\petrtsv\\projects\\ds\\SkyGAN\\trained"
#
# NOISE = 0.1
#
#
# def real_tensor(b_size):
#     noise = NOISE
#     return torch.full((b_size,), real_label) + ((torch.randn((b_size,)) * 2 - 1) * noise)
#
#
# def fake_tensor(b_size):
#     noise = NOISE
#     return torch.full((b_size,), fake_label) + (torch.randn((b_size,)) * noise)
#
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         try:
#             nn.init.normal_(m.weight.data, 0.0, 0.02)
#         except:
#             pass
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#
#
# def pretty_time(seconds):
#     hours = seconds // 3600
#     minutes = seconds // 60 % 60
#     seconds %= 60
#     res = ""
#     if hours:
#         res += '%d h' % hours
#         res += ' '
#         res += '%d m' % minutes
#         res += ' '
#         res += '%d s' % int(seconds)
#     elif minutes:
#         res += '%d m' % minutes
#         res += ' '
#         res += '%d s' % int(seconds)
#     else:
#         res += '%.2f s' % seconds
#     return res
#
#
# FIXED_NOISE_SIZE = 4
# DATALOADER_WORKERS = 2
#
# ADJUST_COEF = 0.99
#
#
# def adjust_learning_rate(optimizer, lr, iteration):
#     cur_lr = lr * (ADJUST_COEF ** (iteration // 1000))
#     for g in optimizer.param_groups:
#         g['lr'] = cur_lr
#
#
# def adversarial_training(epochs,
#                          image_size=256,
#                          batch_size=4,
#                          learning_rate=0.00001,
#                          betas=(0.5, 0.999),
#                          ):
#     random.seed(MANUAL_SEED)
#     torch.manual_seed(MANUAL_SEED)
#     torch.cuda.current_device()
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     generator = ResGenerator(out_channels=1, input_size=nz_size, upsample_blocks_num=4, res_ngf=128).to(device)
#     discriminator = ResDiscriminator(in_channels=1, res_ndf=128).to(device)
#
#     generator.apply(weights_init)
#     discriminator.apply(weights_init)
#
#     optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
#
#     loss = nn.BCELoss()
#
#     train_dataset = CelebaCroppedDataset(transform=scaling_grayscale(image_size))
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                                    shuffle=True, num_workers=DATALOADER_WORKERS)
#
#     adversarial.train(
#         dataloader=train_dataloader,
#         discriminator=discriminator,
#         optimizer_d=optimizer_d,
#         generator=generator,
#         optimizer_g=optimizer_g,
#         epochs=epochs,
#         cur_epoch=0,
#         loss=loss,
#         latent_size=nz_size,
#         noise_level=NOISE,
#     )
#     input("Type ENTER to exit...")
#
#
# if __name__ == '__main__':
#     adversarial_training(epochs=1)

import os

import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import data.celeba
from data.image_transforms import scaling_grayscale
from models.images.generative.gan.dcgan import DCGANGenerator, DCGANDiscriminator
from sessions import Session
from training import adversarial


class SimpleDCGAN(Session):
    def __init__(self,
                 dataset=data.celeba.CelebaCroppedDataset(transform=scaling_grayscale(64)),
                 state_file=None,

                 name="SimpleDCGAN",
                 comment="",
                 ngf=64,
                 ndf=64,
                 latent_size=128,
                 image_size=64,
                 epochs=1,
                 batch_size=256,
                 learning_rate=0.0001,
                 betas=(0.5, 0.999),
                 noise=0.2,
                 dataloader_workers=2,
                 fixed_input_size=4, ):

        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=dataloader_workers)
        self.loss = nn.BCELoss()
        self.generator = DCGANGenerator(
            out_channels=1,
            input_size=latent_size,
            image_size=image_size,
            ngf=ngf).to(self.device)

        self.discriminator = DCGANDiscriminator(
            in_channels=1,
            image_size=image_size,
            ndf=ndf).to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=learning_rate,
                                      betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate,
                                      betas=betas)

        self.build(name=name,
                   comment=comment,
                   state_file=state_file,
                   ngf=ngf,
                   ndf=ndf,
                   latent_size=latent_size,
                   noise=noise,
                   image_size=image_size,
                   epochs=epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   betas=betas,
                   dataloader_workers=dataloader_workers,
                   fixed_input_size=fixed_input_size,
                   )

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            except:
                pass
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def __create__(self, name, comment, **kwargs):
        super(SimpleDCGAN, self).__create__(name, comment, **kwargs)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

    def __restore__(self, data_file):
        super(SimpleDCGAN, self).__restore__(data_file)
        torch_state_file = os.path.join(self.data['checkpoint_dir'], 'torch_state.tar')
        checkpoint = torch.load(torch_state_file)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.train()

        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator.train()

        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        self.loss.load_state_dict(checkpoint['loss_state_dict'])

    def checkpoint(self):
        super(SimpleDCGAN, self).checkpoint()
        torch_state_file = os.path.join(self.data['checkpoint_dir'], 'torch_state.tar')
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
        }, torch_state_file)

    def training(self):
        return adversarial.train(
            session=self,
            dataloader=self.dataloader,
            discriminator=self.discriminator,
            optimizer_d=self.optimizer_d,
            generator=self.generator,
            optimizer_g=self.optimizer_g,
            epochs=self.data['epochs'],
            cur_epoch=0,
            loss=self.loss,
            latent_size=self.data['latent_size'],
            noise_level=self.data['noise'],
        )

    def save_result(self, total_it, g_losses, d_losses, g_progress_list):
        torch.save(self.generator, os.path.join(self.data['output_dir'], "trained_generator_state_dict.tar"))
        torch.save(self.discriminator, os.path.join(self.data['output_dir'], "trained_discriminator_state_dict.tar"))
        iters = list(range(1, total_it + 1))

        plt.figure(figsize=(20, 20))
        plt.plot(iters, g_losses, label="Generator Loss")
        plt.plot(iters, d_losses, label="Discriminator Loss")
        plt.legend()
        plt.savefig(os.path.join(self.data['output_dir'], "loss_plot.png"))

    def run(self):
        self.save_result(
            *self.training()
        )

import time

import numpy as np
import torch
import torchvision.utils as vutils
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sessions import Session
from training import pretty_time
from visualization.image import ImageWindow
from visualization.plots import PlotterWindow


def noisy_tensor(label, size, noise, min_value=None, max_value=None):
    result = torch.full(size, label) + ((torch.rand(size) * 2 - 1) * noise)
    if min_value is not None or max_value is not None:
        result = torch.clamp(result, min=min_value, max=max_value)
    return result


def train(session: Session,
          dataloader: DataLoader,
          generator: nn.Module,
          discriminator: nn.Module,
          optimizer_g: Optimizer,
          optimizer_d: Optimizer,
          loss: nn.Module,
          latent_size: int,
          noise_level: float,
          epochs: int,
          cur_epoch: int = 0,
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          progress_tracker_size=4,
          verbosity_period=50,
          progress_save_period=None,
          ):
    if progress_save_period is None:
        progress_save_period = len(dataloader) // 5
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    loss = loss.to(device)

    plotter = PlotterWindow(interval=1000)
    image_window = ImageWindow(interval=1000)

    it_per_epoch = len(dataloader)

    fixed_noise = torch.randn(progress_tracker_size ** 2, latent_size, 1, 1, device=device)

    session.checkpoint()
    session.info['status'] = 'Running'
    print("Adversarial training started\n")

    start_time = time.time()

    cur_it = it_per_epoch * cur_epoch
    total_it = it_per_epoch * epochs

    plotter.new_line('g_losses')
    g_losses = []

    plotter.new_line('d_losses')
    d_losses = []
    g_progress_list = []
    for epoch in range(cur_epoch, epochs):
        for epoch_it, data in enumerate(dataloader, 0):
            cur_batch_size = data.size(0)

            discriminator.zero_grad()

            # Real batch
            real = data.to(device)
            label = noisy_tensor(1, (cur_batch_size,), noise_level).to(device)
            output = discriminator(real).view(-1)

            loss_d_real = loss(output, label)
            loss_d_real.backward()
            d_x = output.mean().item()

            # Fake batch
            noise = torch.randn(cur_batch_size, latent_size, 1, 1, device=device)
            fake = generator(noise)
            label = noisy_tensor(0, (cur_batch_size,), noise_level, min_value=0).to(device)
            output = discriminator(fake.detach()).view(-1)

            loss_d_fake = loss(output, label)
            loss_d_fake.backward()
            d_g_z = output.mean().item()
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()

            # Generator
            generator.zero_grad()
            output = discriminator(fake).view(-1)

            label = torch.full((cur_batch_size,), 1, device=device)
            loss_g = loss(output, label)
            loss_g.backward()
            optimizer_g.step()

            cur_it += 1

            cur_time = time.time()
            delta_time = cur_time - start_time
            time_per_it = delta_time / cur_it

            plotter.add_point('g_losses', cur_it, loss_g.item())
            g_losses.append(loss_g.item())
            plotter.add_point('d_losses', cur_it, loss_d.item())
            d_losses.append(loss_d.item())

            with torch.no_grad():
                cur_fake = generator(fixed_noise).detach().cpu()
            cur_generated = vutils.make_grid(cur_fake, padding=2, normalize=True, nrow=progress_tracker_size)
            image_window.set_image(np.transpose(cur_generated, (1, 2, 0)))

            if epoch_it % verbosity_period == 0 or epoch_it == len(dataloader) - 1:
                print('[%d/%d][%d/%d] = %.2f%%\t\tLoss_G: %.4f\tLoss_D: %.4f\tD(y): %.4f\tD(G(x)): %.4f' %
                      (epoch + 1, epochs, epoch_it, it_per_epoch, cur_it / total_it * 100,
                       loss_g.item(), loss_d.item(), d_x, d_g_z))
                print('Average iteration time: %s\tAverage epoch time: %s\tEstimated execution time: %s' % (
                    pretty_time(time_per_it),
                    pretty_time(time_per_it * it_per_epoch),
                    pretty_time(time_per_it * (it_per_epoch * epochs - cur_it)),
                ))
                print()
            if epoch_it % progress_save_period == 0 or epoch_it == len(dataloader) - 1:
                g_progress_list.append(cur_generated)

        session.checkpoint()

    cur_time = time.time()
    delta_time = cur_time - start_time
    session.info['status'] = "Finished"
    print("Training finished. Total execution time: %s" % pretty_time(delta_time))
    print()
    return total_it, g_losses, d_losses, g_progress_list

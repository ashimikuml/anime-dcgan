from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from .model import Generator, Discriminator


def train(dataset, max_iter, ckpt_path, save_iter=5000, lr=0.0002, batch_size=64, manual_seed=None, cuda=True, resume=True):
    manual_seed = None
    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    cudnn.benchmark = True
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    nz = 100
    netG = Generator(nz=nz)
    netD = Discriminator()
    criterion = nn.BCELoss()

    if cuda:
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    netG.to(device)
    netD.to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    start_iter = 0

    writer = SummaryWriter()

    if resume and os.path.exists(ckpt_path):
        # ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = torch.load(ckpt_path)
        start_iter = ckpt['iteration']
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        optimizerG.load_state_dict(ckpt['optimizerG'])
        optimizerD.load_state_dict(ckpt['optimizerD'])
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    dataloader_iter = iter(dataloader)
    for iteration in range(start_iter, max_iter):
        try:
            data = dataloader_iter.next()
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = dataloader_iter.next()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        label = torch.full((real_cpu.size(0),), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(real_cpu.size(0), nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if iteration % 200 == 0:
            print('%d/%d errD_real:%.2e errD_fake:%.2e errG:%.2e \r' % (iteration, max_iter, errD_real.item(), errD_fake.item(), errG.item()), end='')

        if iteration % 1000 == 0:
            writer.add_scalar('anime-dcgan/errD_real', errD_real.item(), iteration)
            writer.add_scalar('anime-dcgan/errD_fake', errD_fake.item(), iteration)
            writer.add_scalar('anime-dcgan/errG', errG.item(), iteration)

            netG.eval()
            with torch.no_grad():
              grid = vutils.make_grid((real_cpu * 255).byte(), range=(0, 255), scale_each=True)
              writer.add_image('real', grid, iteration)
              fake = netG(fixed_noise)
              grid = vutils.make_grid((fake * 255).byte(), range=(0, 255), scale_each=True)
              writer.add_image('fixed_fake', grid, iteration)
            netG.train()


        if iteration > 0 and iteration % save_iter == 0:
            save(netD, netG, optimizerD, optimizerG, iteration, ckpt_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save(netD, netG, optimizerD, optimizerG, iteration, ckpt_path):
     torch.save({
        'iteration': iteration,
        'netD': netD.state_dict(),
        'netG': netG.state_dict(),
        'optimizerD' : optimizerD.state_dict(),
        'optimizerG' : optimizerG.state_dict(),
    }, ckpt_path)

import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from models import *
from datasets import *

from matplotlib import pyplot as plt

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs("images/training", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # GPU or CPU (Not having at least 1 GPU can prevent code from working)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device: ", device)

    # Define parameters
    dataroot = f"D:\Datasets\hands_v4\Hands\Hands"

    ### PARAMS

    epoch=0 # epoch to start training from
    n_epochs=200 # number of epochs of training
    batch_size=16 # size of the batches (default = 4), 16 occupies 23.5 GB of vram
    lr=0.0002 # adam: learning rate
    b1=0.9 # adam: decay of first order momentum of gradient
    b2=0.999 # adam: decay of first order momentum of gradient
    decay_epoch=100 # epoch from which to start lr decay
    n_cpu=8 # number of cpu threads to use during batch generation
    hr_height=256 # high res. image height
    hr_width=256 # high res. image width
    channels=3 # number of image channels
    sample_interval=100 # nterval between saving image samples
    checkpoint_interval=1000 #batch interval between model checkpoints
    residual_blocks=23 # number of residual blocks in the generator
    warmup_batches=100 # number of batches with pixel-wise loss only (default = 500)
    lambda_adv=5e-3 # adversarial loss weight
    lambda_pixel=1e-2 # pixel-wise loss weight

    hr_shape = (hr_height, hr_width)

    ###

    # Initialize generator and discriminator
    generator = GeneratorRRDB(channels, filters=64, num_res_blocks=residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    print(generator)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    if epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    dataloader = DataLoader(
        ImageDataset(dataroot, hr_shape=hr_shape),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    print(f'Number of images: {len(dataloader) * batch_size}')

    # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Flags - For each epoch
    show_images = True
    save_images = True
    save_model = True

    # ----------
    #  Training
    # ----------

    for epoch in range(epoch, n_epochs):
        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "🔥 [Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, n_epochs, i, len(dataloader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )

            if batches_done % sample_interval == 0:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
                save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

            if batches_done % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
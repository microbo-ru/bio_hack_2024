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

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define parameters
    dataroot = "/home/vladimir/Work/microbo/hands_prep/Hands/Hands/"
    nb_channels = 3 # For RGB images, but if you use grayscale images, ToTensor() will replicate the single channel into three channels, so you should not have to modify anything
    image_resize = 64

    batch_size = 128
    nb_gpu = 1
    nb_workers = 4 # based on system resources

    # GPU or CPU (Not having at least 1 GPU can prevent code from working)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and nb_gpu > 0) else "cpu")

    # Create the dataset by applying transformation to our images
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_resize),
                                transforms.CenterCrop(image_resize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    
    print(f'Number of downloaded images: {len(dataset)}')
    # exit()

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=nb_workers)

    real_batch = next(iter(dataloader))

    nb_images = 9
    nb_row = math.ceil(math.sqrt(nb_images))

    # Size of z latent vector (i.e. size of generator input), same size as described in the DCGAN paper
    nz = 128
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 100
    # Learning rate for optimizers, same value as described in the DCGAN paper
    lr = 0.0002
    # Beta1 hyperparameter for Adam optimizers, same value as described in the DCGAN paper
    beta1 = 0.5

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code
    class Generator(nn.Module):
        def __init__(self, nb_gpu):
            super(Generator, self).__init__()
            self.nb_gpu = nb_gpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                
                nn.ConvTranspose2d(ngf, nb_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nb_channels) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)

    # Create the generator
    netG = Generator(nb_gpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (nb_gpu > 1):
        netG = nn.DataParallel(netG, list(range(nb_gpu)))

    # Apply the ``weights_init`` funb_channelstion to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    # exit()

    class Discriminator(nn.Module):
        def __init__(self, nb_gpu):
            super(Discriminator, self).__init__()
            self.nb_gpu = nb_gpu
            self.main = nn.Sequential(
                # input is ``(nb_channels) x 64 x 64``
                
                nn.Conv2d(nb_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)
        
    # Create the Discriminator
    netD = Discriminator(nb_gpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (nb_gpu > 1):
        netD = nn.DataParallel(netD, list(range(nb_gpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Set real and fake label values (following GAN paper convention)
    real_label = 1.
    fake_label = 0.

    # Define loss function
    criterion = nn.BCELoss()

    # Setup optimizers for both G and D, according to the DCGAN paper
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Flags - For each epoch
    show_images = False
    save_images = False
    save_model = True

    def save_dcgan(netG, netD, path_checkpoint):
        checkpoint = {"g_model_state_dict": netG.state_dict(),
                    "d_model_state_dict": netD.state_dict(),
                    }
        
        torch.save(checkpoint, path_checkpoint)
        
    def makedir(new_dir):
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    # Create folders
    makedir("images")
    makedir("models")

    # Training Loop
    data_len = len(dataloader)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []


    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader (depends on batch_size and your number of images)
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            ############################
            # (3) Metrics & Evaluation
            ###########################
            
            # Print output training stats every 50 batches (if your dataset is large, printing at every epoch might be less frequent than you want)
            if i % 50 == 0:
                print(f"Epoch: {epoch}/{num_epochs} Batches: {i}/{data_len}\tLoss_D: {errD.item():.4f}   Loss_G: {errG.item():.4f}    D(x): {D_x:.4f}    D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")
                
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
                
        # Generate fake images to see how the generator is doing by saving G's output on fixed_noise at each epoch (fixed noise allow to obtain similar images).
        # if show_images == True:
        #     with torch.no_grad():
        #         # Uncomment the line below to generate a new variety of images every time
        #         #fixed_noise = torch.randn(64, nz, 1, 1, device=device)
                
        #         fake = netG(fixed_noise).detach().cpu()
        #         img_list.append(vutils.make_grid(fake[:nb_images], padding=2, normalize=True, nrow=nb_row))

        #         plt.figure(figsize=(3, 3))
        #         plt.axis("off")
        #         plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
                
        #         if save_images == True:
        #             plt.savefig(f'images/epoch_{epoch}_gen_images.png')
                    
        #         # Display image  
        #         plt.show()
        
        # Save models each 5 epochs
        if epoch % 5 == 0:
            if save_model:
                save_dcgan(netG, netD, path_checkpoint=f"models/hands_epoch_{epoch}_checkpoint.pkl")
            
    # Save the final models
    save_dcgan(netG, netD, path_checkpoint="models/hands_final_epoch_checkpoint.pkl")
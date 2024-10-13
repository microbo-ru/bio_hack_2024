import argparse
import math
import torch
import torch.nn as nn
from pathlib import Path
from RealESRGAN import RealESRGAN
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
import shutil
from tqdm import tqdm

# python inference.py -n 10 -o out

parser = argparse.ArgumentParser(prog='example')
parser.add_argument('-m', '--model', nargs='?', default='weights/hands_final_epoch_checkpoint.pkl', help='Input model')
parser.add_argument('-n', '--number', required=True, type=int, help='Number of images to generate')
parser.add_argument('-o', '--output', required=True, help='Output folder for images')

nb_gpu = 1
nb_channels = 3 
nz = 128
ngf = 64
ndf = 64

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

if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available() and nb_gpu > 0) else "cpu")

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    path_checkpoint = args.model
    
    t_weights = Path(path_checkpoint)
    if not t_weights.is_file():
        print("Please load weights for palmar GAN. see Readme")
        exit()

    num_img = args.number
    nb_row = math.ceil(math.sqrt(num_img))

    # Create a random noise
    random_noise = torch.randn(num_img, nz, 1, 1, device=device)

    # Instantiate a generator
    new_gen= Generator(nb_gpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (nb_gpu > 1):
        new_gen = nn.DataParallel(new_gen, list(range(nb_gpu)))

    # Load weights from path
    checkpoint = torch.load(path_checkpoint, map_location="cpu")
    state_dict_g = checkpoint["g_model_state_dict"]
    new_gen.load_state_dict(state_dict_g)

    # Generate images
    with torch.no_grad():
        fake_data = new_gen(random_noise).detach().cpu()

    # RealESRGAN
    t_weights = Path('weights/RealESRGAN_x8.pth')
    if not t_weights.is_file():
        print("Please load weights for RealESRGAN. see Readme")
        exit()

    scale_model = RealESRGAN(device, scale=8)
    scale_model.load_weights('weights/RealESRGAN_x8.pth', download=False)

    tmp_out = f'{args.output}_64'
    shutil.rmtree(Path(f'{tmp_out}/'))
    Path(tmp_out).mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(len(fake_data))):
        save_image(fake_data[idx], f'{tmp_out}/{idx}.png')
        image = Image.open(f'{tmp_out}/{idx}.png').convert('RGB')
        sr_image = scale_model.predict(image)
        sr_image.save(f'{args.output}/{idx}.png')


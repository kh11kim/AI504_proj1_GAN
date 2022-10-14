import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import imageio
import numpy as np
# torch
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dcgan.network import Discriminator, Generator

CHECKPOINT_PATH = './checkpoint'
DATASET_PATH = './dataset'
IMG_PATH = './img'
REAL_IMG_PATH = './img/real'
FAKE_IMG_PATH = './img/fake'

def create_folders():
    path_list = [
        CHECKPOINT_PATH,
        DATASET_PATH,
        IMG_PATH,
        REAL_IMG_PATH,
        FAKE_IMG_PATH]
    
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

def get_dataset():
    pass

def show_image(image:torch.Tensor):
    if len(image.shape) > 2:
        image = image[0]
    plt.imshow(image.detach().cpu().numpy(), cmap="gray")
    plt.show()

def save_gif(training_progress_images, images):
    '''
        training_progress_images: list of training images generated each iteration
        images: image that is generated in this iteration
    '''
    img_grid = make_grid(images.data)
    img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
    img_grid = 255. * img_grid 
    img_grid = img_grid.astype(np.uint8)
    training_progress_images.append(img_grid)
    imageio.mimsave('./img/training_progress.gif', training_progress_images)
    return training_progress_images

def save_image_list(dataset, real):
    if real:
        base_path = REAL_IMG_PATH
    else:
        base_path = FAKE_IMG_PATH
    
    dataset_path = []
    
    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image(dataset[i], save_path)
    
    return base_path

def main():
    # set parameters
    lr = 0.0002
    n_epoch = 200
    batch_size = 128

    create_folders()
    writer = SummaryWriter()
    # prepare dataset
    transform_pipeline = transforms.Compose([transforms.ToTensor(),])
    dataset = dset.MNIST(
        root=DATASET_PATH, 
        download=True, 
        transform=transform_pipeline)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # network
    netD = Discriminator(784).cuda() #in: image vector (784,)
    netG = Generator(100, 784).cuda() #in: random vector (100,), out: 784(28*28) vector
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # training loop
    print("Training start")
    fixed_noise = torch.randn(128, 100).cuda()
    
    progress_image_list = []
    for epoch in range(n_epoch):
        for i, (data, _) in enumerate(dataloader):
            batch_size = data.shape[0]
            data = data.cuda()

            ## 1) update netD
            netD.zero_grad()
            # train with real
            label = torch.ones(batch_size).cuda()
            pred: torch.Tensor = netD(data)
            errD_real = criterion(pred, label)
            D_x = pred.mean().item()
            # train with fake
            noise = torch.randn(batch_size, 100).cuda()
            fake: torch.Tensor = netG(noise) 
            label = torch.zeros(batch_size).cuda()
            pred = netD(fake.detach()) # to prevent calculating gradient from netG
            errD_fake = criterion(pred, label)
            D_G_z1 = pred.mean().item()
            # update
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            ## 2) update netG
            netG.zero_grad()
            label = torch.ones(batch_size).cuda()
            pred = netD(fake)
            errG = criterion(pred, label)
            D_G_z2 = pred.mean().item()
            # update
            errG.backward()
            optimizerG.step()
        
        print(f'[{epoch}/{n_epoch}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
        writer.add_scalars("losses/g-and-d", {'g':errG.item(), "d":errD.item()}, epoch)
        with torch.no_grad():
            gen_images = netG(fixed_noise)
            gen_images_resized = nn.Upsample(scale_factor=1.5, mode='nearest')(gen_images)
            grid = make_grid(gen_images_resized, nrow=8, normalize=True)
            writer.add_image("intermediate fake", grid, epoch)

        #save the output
        #fake = netG(fixed_noise)
        #training_progress_images_list = save_gif(training_progress_images_list, fake)  # Save fake image while training!
        # Check pointing for every epoch
        torch.save(netG.state_dict(), f'./checkpoint/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'./checkpoint/netD_epoch_{epoch}.pth')


    #show_image(dataset.data[0])
    print("a")

if __name__ == "__main__":
    main()
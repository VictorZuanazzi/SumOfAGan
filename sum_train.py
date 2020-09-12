"""Implemented by Victor Zuanazzi"""

import argparse
import os
from os.path import join

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST

from tqdm import tqdm

from cgan import GeneratorMLP, DiscriminatorMLP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="exp",
                        help="Name of the experiment. Logs and Models will "
                             "be saved under this name.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs.")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="number of dimensions of the latent input.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size: number of simultaneous examples "
                             "per training step.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate used by the optimizer.")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="Momentum contribution of the optimizer.")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="decay of second order gradient decay.")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="number of cpu threads for batch loading.")
    parser.add_argument("--img_size", type=int, default=32,
                        help="The Width and Height of an squared image.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Uses the code in debug mode.")

    config = parser.parse_args()

    # initilize logger
    os.makedirs(config.exp_name, exist_ok=True)
    writer = SummaryWriter(join(config.exp_name, "log"))

    # checkpoint paths
    model_path = join(config.exp_name, "model")
    os.makedirs(model_path, exist_ok=True)
    img_path = join(config.exp_name, "images")
    os.makedirs(img_path, exist_ok=True)

    # initialize dataset and dataloader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "../data/"
    os.makedirs(data_path, exist_ok=True)

    dataset = MNIST(data_path, train=True, download=True,
                    transform=transforms.Compose(
                        [transforms.Resize(config.img_size),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5]),
                         ]))
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size,
                            shuffle=True)
    n_classes = 10
    image_shape = (1, config.img_size, config.img_size)

    # load models
    generator = GeneratorMLP(n_classes=n_classes,
                             latent_dim=config.z_dim,
                             img_shape=image_shape)
    discriminator = DiscriminatorMLP(n_classes, img_shape=image_shape)
    generator.to(device)
    discriminator.to(device)

    # initialize optimizers
    opt_g = Adam(generator.parameters(),
                 lr=config.lr, betas=(config.b1, config.b2))
    opt_d = Adam(discriminator.parameters(),
                 lr=config.lr, betas=(config.b1, config.b2))
    loss_func = nn.BCEWithLogitsLoss()

    z_fixed = torch.randn((n_classes * 10, config.z_dim), device=device)
    label_fixed = torch.arange(n_classes, device=device).repeat(10)

    for epoch in range(config.epochs):
        for i, (img_real, label) in enumerate(tqdm(dataloader)):

            img_real = img_real.to(device)
            label = label.to(device)
            b = label.shape[0]

            # Train Discriminator:
            discriminator.train()
            generator.eval()
            opt_d.zero_grad()

            # real images
            score_real = discriminator(img_real, label)
            loss_real_d = loss_func(score_real, torch.ones_like(score_real))

            # fake images
            z = torch.randn(size=(b, config.z_dim), device=device)
            img_fake = generator(z, label)
            score_fake = discriminator(img_fake.detach(), label)
            loss_fake_d = loss_func(score_fake, torch.zeros_like(score_fake))

            loss_d = (loss_fake_d + loss_real_d) / 2
            loss_d.backward()
            opt_d.step()

            # Train Generator:
            discriminator.eval()
            generator.train()
            opt_g.zero_grad()

            z = torch.randn(size=(b, config.z_dim), device=device)
            img_fake = generator(z, label)
            score_fake = discriminator(img_fake, label)
            loss_g = loss_func(score_fake, torch.zeros_like(score_fake))

            loss_g.backward()
            opt_g.step()

            if config.debug:
                break

        # log last epoch of training
        writer.add_scalars(main_tag="loss",
                           tag_scalar_dict={"D": loss_d.item(),
                                            "G": loss_g.item()},
                           global_step=epoch)

        img_training = torch.cat((make_grid(img_fake, nrow=1, normalize=True),
                                  make_grid(img_real, nrow=1, normalize=True)),
                                 dim=-1)
        writer.add_image(tag="training", img_tensor=img_training,
                         global_step=epoch)

        # generate images per class
        generator.eval()
        img_class = make_grid(generator(z_fixed, label_fixed),
                              nrow=n_classes, normalize=True)
        writer.add_image(tag="classes", img_tensor=img_class,
                         global_step=epoch)
        save_image(img_class, join(img_path, f"fake_{epoch}.png"))

        # chekcpoint models:
        torch.save(generator.state_dict(),
                   join(model_path, f"generator_{epoch}.pt"))
        torch.save(discriminator.state_dict(),
                   join(model_path, f"discriminator_{epoch}.pt"))

        if config.debug:
            break

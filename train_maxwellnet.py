# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from Dataset import LensDataset
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from MaxwellNet import MaxwellNet

import numpy as np
import random
import logging
import argparse
import os
import json
from datetime import datetime


def main(directory, load_ckpt):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(
                            os.getcwd(), directory, f"maxwellnet_{datetime.now():%Y-%m-%d %H-%M-%S}.log"),
                        filemode='w')

    logging.info("training " + directory)

    specs_filename = os.path.join(directory, 'specs_maxwell.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs_maxwell.json"'
        )

    specs = json.load(open(specs_filename))

    seed_number = get_spec_with_default(specs, "Seed", None)
    if seed_number != None:
        fix_seed(seed_number, torch.cuda.is_available())

    rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Experiment description: \n" +
                 ' '.join([str(elem) for elem in specs["Description"]]))
    logging.info("Training with " + str(device))

    model = MaxwellNet(**specs["NetworkSpecs"], **specs["PhysicalSpecs"])
    if torch.cuda.device_count() > 1:
        logging.info("Multiple GPUs: " + str(torch.cuda.device_count()))
    if load_ckpt is not None:
        load_path = os.path.join(os.getcwd(), directory, 'model', load_ckpt)
        ckpt_dict = torch.load(load_path + '.pt')
        ckpt_epoch = ckpt_dict['epoch']
        logging.info("Checkpoint loaded from {}-epoch".format(ckpt_epoch))
        model.load_state_dict(ckpt_dict['state_dict'])

    model = torch.nn.DataParallel(model)
    model.train()
    model = model.to(device)

    logging.info("Number of network parameters: {}".format(
        sum(p.data.nelement() for p in model.parameters())))
    logging.debug(specs["NetworkSpecs"])
    logging.debug(specs["PhysicalSpecs"])

    optimizer = torch.optim.Adam(model.parameters(), lr=get_spec_with_default(
        specs, "LearningRate", 0.0001), weight_decay=0)
    scheduler = StepLR(optimizer, step_size=get_spec_with_default(
        specs, "LearningRateDecayStep", 10000), gamma=get_spec_with_default(specs, "LearningRateDecay", 1.0))

    batch_size = get_spec_with_default(specs, "BatchSize", 1)
    epochs = get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]
    physical_specs = specs["PhysicalSpecs"]
    symmetry_x = physical_specs['symmetry_x']
    mode = physical_specs['mode']
    high_order = physical_specs['high_order']

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode + '_' + high_order
    writer = SummaryWriter(os.path.join(directory, 'tensorboard_' + filename))
    writer_freq = get_spec_with_default(specs, "TensorboardFrequency", None)

    train_dataset = LensDataset(directory, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, sampler=None)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    loss_train = torch.zeros(
        (int(epochs),), dtype=torch.float32, requires_grad=False)

    if len(train_dataset) > 1:
        perform_valid = True
    else:
        perform_valid = False

    if perform_valid == True:
        valid_dataset = LensDataset(directory, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True, sampler=None)
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = torch.zeros(
            (int(epochs),), dtype=torch.float32, requires_grad=False)

    if load_ckpt is not None:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        loss_train[:ckpt_epoch:] = ckpt_dict['loss_train'][:ckpt_epoch:]
        logging.info("Check point loaded from {}-epoch".format(ckpt_epoch))

        start_epoch = ckpt_epoch
    else:
        start_epoch = 0

    logging.info("Training start")

    for epoch in range(start_epoch + 1, epochs + 1):
        train(train_loader, model, optimizer, epoch, loss_train,
              device, mode, symmetry_x, writer, writer_freq)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, loss_train[epoch-1].item())) if rank == 0 else None
        if perform_valid:
            valid(valid_loader, model, epoch, loss_valid,
                  device, mode, symmetry_x, writer, writer_freq)
            logging.info("[Valid] {} epoch. Loss: {:.5f}".format(
                epoch, loss_valid[epoch-1].item())) if rank == 0 else None

        if epoch in checkpoints:
            logging.info("Checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                }, directory, str(epoch) + '_' + mode + '_' + high_order)

        if epoch % 200 == 0:
            logging.info("'latest' checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                }, directory, 'latest')

        scheduler.step()

    writer.close() if rank == 0 else None


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, symmetry, writer, writer_freq):
    model.train()
    with torch.set_grad_enabled(True):
        count = 0

        for data in train_loader:
            scat_pot_torch = data[0].to(device)
            ri_value_torch = data[1].to(device)

            (diff, total) = model(scat_pot_torch, ri_value_torch)

            l2 = diff.pow(2)
            loss = torch.mean(l2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            loss_train[epoch-1] += loss.item() * diff.size(0)
            count += diff.size(0)

        loss_train[epoch-1] = loss_train[epoch-1] / count

    if epoch % writer_freq == 0 and writer != None:
        to_tensorboard(total.permute(2, 3, 1, 0)[:, :, :, 0].clone().detach().cpu(), loss_train[epoch-1].numpy(), epoch,
                       mode, symmetry, writer, 'train')


def valid(valid_loader, model, epoch, loss_valid, device, mode, symmetry, writer, writer_freq):
    model.eval()
    with torch.set_grad_enabled(False):
        count = 0

        for data in valid_loader:
            scat_pot_torch = data[0].to(device)
            ri_value_torch = data[1].to(device)

            (diff, total) = model(scat_pot_torch,
                                  ri_value_torch)  # [N, 1, H, W, D]

            l2 = diff.pow(2)
            loss = torch.mean(l2)

            loss_valid[epoch-1] += loss.item() * diff.size(0)
            count += diff.size(0)

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    if epoch % writer_freq == 0 and writer != None:
        to_tensorboard(total.permute(2, 3, 1, 0)[:, :, :, 0].clone().detach().cpu(), loss_valid[epoch-1].numpy(), epoch,
                       mode, symmetry, writer, 'valid')


def to_tensorboard(image, losses, epoch, mode, symmetry, writer, train_valid):
    if symmetry is True:
        if mode == 'te':
            y_pol = torch.cat((torch.flip(image, [0])[0:-1:, :, :], image), 0)
        elif mode == 'tm':
            x_pol = torch.cat((torch.flip(image, [0]), image), 0)
            z_pol = torch.cat((-torch.flip(image, [0])[0:-1, :, :], image), 0)

    if mode == 'te':
        polarization = ['y']
    elif mode == 'tm':
        polarization = ['x', 'z']

    for idx in range(len(polarization)):
        if symmetry == True:
            if polarization[idx] == 'y':
                image = y_pol
            elif polarization[idx] == 'x':
                image = x_pol
            elif polarization[idx] == 'z':
                image = z_pol

        amplitude = torch.sum(
            image[:, :, idx*2:(idx+1)*2].pow(2), 2).pow(1 / 2)
        amplitude = amplitude - torch.min(amplitude)
        amplitude = amplitude / torch.max(amplitude)
        writer.add_image(train_valid + '/' + mode + '/amplitude_' +
                         polarization[idx], amplitude.unsqueeze(0), epoch)

        real = image[:, :, idx*2]
        real = real - torch.min(real)
        real = real / torch.max(real)
        writer.add_image(train_valid + '/' + mode + '/real_' +
                         polarization[idx], real.unsqueeze(0), epoch)

        imaginary = image[:, :, idx*2+1]
        imaginary = imaginary - torch.min(imaginary)
        imaginary = imaginary / torch.max(imaginary)
        writer.add_image(train_valid + '/' + mode + '/imaginary_' +
                         polarization[idx], imaginary.unsqueeze(0), epoch)

    writer.add_scalar(train_valid + '/' + mode, losses, epoch)


def save_checkpoint(state, directory, filename):
    model_directory = os.path.join(directory, 'model')
    if os.path.exists(model_directory) == False:
        os.makedirs(model_directory)
    torch.save(state, os.path.join(model_directory, filename + '.pt'))


def fix_seed(seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--directory",
        "-d",
        required=True,
        default='examples\spheric_te',
        help="This directory should include "
             + "all the training and network parameters in 'specs_maxwell.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--load_ckpt",
        "-l",
        default=None,
        help="This should specify a filename of your checkpoint within 'directory'\model if you want to continue your training from the checkpoint.",
    )

    args = arg_parser.parse_args()
    main(args.directory, args.load_ckpt)

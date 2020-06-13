import os
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF

from torch.utils.tensorboard import SummaryWriter

from core.text_detection import TextDetection as TD
from data_loader import SynthData
from utils.toolkit import craft_dict_adapt
from native_craft.craft import CRAFT

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    for i, (images, region_gt, affinity_gt) in enumerate(train_loader):

        images = images.cuda()
        region_gt = region_gt.cuda()
        affinity_gt = affinity_gt.cuda()
        #mask = mask.cuda()

        #out = net(images)
        out, _ = net(images)
        optimizer.zero_grad()

        #region_pdt = out[:, 0, :, :].cuda()
        #affinity_pdt = out[:, 1, :, :].cuda()
        
        region_pdt = out[:, :, :, 0].cuda()
        affinity_pdt = out[:, :, :, 1].cuda()

        loss = (torch.mean(criterion(region_pdt, region_gt)) + torch.mean(criterion(affinity_pdt, affinity_gt))) / batch_size
        #loss = criterion(region_gt, affinity_gt, region_pdt, affinity_pdt, mask)
        loss.backward()

        optimizer.step()
        loss_value += loss.item()
        if (i + 1) % config.PRINT_FREQ == 0:

            loss_value /= config.PRINT_FREQ
            print('batch {1}/{2} \tloss {loss.val:.5f}'.format(epoch, i + 1, len(train_loader), loss=loss_value))
            loss_value = 0

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
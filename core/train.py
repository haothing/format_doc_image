import os
import shutil
import random
import torch

def train(config, train_loader, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    loss_value = 0
    for i, (images, region_gt, affinity_gt) in enumerate(train_loader):

        images = images.to(device)
        region_gt = region_gt.to(device)
        affinity_gt = affinity_gt.to(device)
        #mask = mask.cuda()

        out, _ = model(images)
        optimizer.zero_grad()

        #region_pdt = out[:, 0, :, :].cuda()
        #affinity_pdt = out[:, 1, :, :].cuda()
        
        region_pdt = out[:, :, :, 0].to(device)
        affinity_pdt = out[:, :, :, 1].to(device)

        loss = (torch.mean(criterion(region_pdt, region_gt)) + torch.mean(criterion(affinity_pdt, affinity_gt))) / config.TRAIN.BATCH_SIZE_PER_GPU
        #loss = criterion(region_gt, affinity_gt, region_pdt, affinity_pdt, mask)
        loss.backward()

        optimizer.step()
        loss_value += loss.item()
        if (i + 1) % config.PRINT_FREQ == 0:

            loss_value /= config.PRINT_FREQ
            print('batch {}/{} \tloss {:.8f}'.format(i + 1, len(train_loader), loss_value))

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', loss_value, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
            
            loss_value = 0
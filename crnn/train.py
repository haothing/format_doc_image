import argparse
from easydict import EasyDict as edict
import yaml
import os
import time, datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets

import sys 
sys.path.append('../')
from core.transforms import RandomCompressTransform

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    #config.DATASET.ALPHABETS = alphabets.alphabet

    # use japanese character
    char_file = open(os.path.join('./lib/config/', config.DATASET.ALPHABETS), "r", encoding="utf-8")
    config.DATASET.ALPHABETS = ''.join(char_file.read().splitlines())
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def setup_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_dist():
    dist.destroy_process_group()

def main():

    # load config
    config = parse_arg()
    config.DISTRIBUTER = config.DISTRIBUTER and torch.distributed.is_available()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if config.DISTRIBUTER:
        setup_dist(config.RANK, config.WORLD_SIZE)
    
    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        if config.DISTRIBUTER:
            device = config.RANK
        else:
            device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)
    if config.DISTRIBUTER:
        model = DDP(model, device_ids=[device])

    # define loss function
    criterion = torch.nn.CTCLoss()

    optimizer = utils.get_optimizer(config, model)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR
        )

    tc = transforms.Compose([
        RandomCompressTransform(scale_width=(0.6, 1)),
        transforms.RandomPerspective(distortion_scale =0.1),
        transforms.RandomRotation(3),
        transforms.ToTensor()])

    train_dataset = get_dataset(config)(config, is_train=True, transform=tc)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False, transform=tc)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    start_time = time.time()
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        
        print('epoch: {}/{} \telapsed time: {}\tlearning rate: {}'.format(epoch + 1, config.TRAIN.END_EPOCH,
            str(datetime.timedelta(seconds=(time.time() - start_time) // 1)), lr_scheduler.get_last_lr()))
        loss = function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        if config.TRAIN.CHECK_EPOCH > 0 and epoch + 1 == last_epoch + config.TRAIN.CHECK_EPOCH:
            #if loss < config.TRAIN.CHECK_LOSS:
            torch.save(
                {"state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": loss,
                }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_loss_{:.4f}.pth".format(epoch + 1, loss))
            )
            break

        lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        #print("is best:", is_best)
        #print("best acc is:", best_acc)
        # save checkpoint
        if (epoch + 1) % config.SAVE_CP_FREQ == 0 and (epoch + 1) >= config.SAVE_CP_FROM:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch + 1, acc))
            )

    writer_dict['writer'].close()

    if config.DISTRIBUTER:
        cleanup_dist()

if __name__ == '__main__':
    for i in range(10):
        main()
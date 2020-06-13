import argparse
from easydict import EasyDict as edict
import yaml
import os
import time, datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils

import core.trian
from data_loader import SynthData

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from core.transforms import RotationListTransform, ToTensorListTransform
import utils.toolkit as toolkit
from core.text_detection import TextDetection as TD

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
    output_dict = toolkit.create_log_folder(config, phase='train')

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

    # get device
    if torch.cuda.is_available():
        if config.DISTRIBUTER:
            device = config.RANK
        else:
            device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = toolkit.craft_dict_adapt(torch.load(model_state_file))
        model = CRAFT(freeze=True)
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']
    else:
        model = CRAFT(pretrained=True, freeze=False)

    if config.DISTRIBUTER:
        setup_dist(config.RANK, config.WORLD_SIZE)
        model = DDP(model, device_ids=[device])
    else:
        model = model.to(device)
    criterion = torch.nn.CTCLoss()
    optimizer = toolkit.get_optimizer(config, model)
    
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    obj_transforms = transforms.Compose([RotationListTransform(angles=[0, 30, 60, 90, -30, -60, -90]), ToTensorListTransform()])
    train_dataset = SynthData(target_size=config.MODEL.IMAGE_SIZE, transform=obj_transforms, root=config.DATASET.ROOT, ground_true_file=config.DATASET.JSON_FILE['train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    test_images_folder = './test/images'
    start_time = time.time()
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        
        print('start epoch: {}/{} \telapsed time: {}\tlearning rate: {}'.format(epoch, config.TRAIN.END_EPOCH,
            str(datetime.timedelta(seconds=(time.time() - start_time) // 1)), lr_scheduler.get_last_lr()))
        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        #acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        checkpoint_filename = "checkpoint_epoch_{}.pth".format(epoch)
        torch.save({"state_dict": model.state_dict(),"epoch": epoch + 1}, os.path.join(output_dict['chs_dir'], checkpoint_filename))

        text_dtc = TD(weight_path=checkpoint_filename)
        text_dtc.draw_to_file(test_images_folder, dist_dir=os.path.join(test_images_folder, 'result', 'epoch_' + str(epoch)), heatmap=True)    

    writer_dict['writer'].close()

    if config.DISTRIBUTER:
        cleanup_dist()

if __name__ == '__main__':
    for i in range(10):
        main()
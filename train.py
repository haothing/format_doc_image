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

class RotationListTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        assert type(x) == list, 'Only list can be passed as argument.'
        angle = random.choice(self.angles)
        return [TF.rotate(item, angle) for item in x]

class ToTensorListTransform:
    def __call__(self, x):
        assert type(x) == list, 'Only list can be passed as argument.'
        
        t = transforms.ToTensor()
        return [t(item) for item in x]

if __name__ == '__main__':

    obj_transforms = transforms.Compose([RotationListTransform(angles=[0, 30, 60, 90, -30, -60, -90]), ToTensorListTransform()])
    train_dataset = SynthData("E:/datasets/SynthText/SynthText_Gen", target_size=768, transform=obj_transforms, ground_true_file='from_ICDAR2017.mat')
    num_folder = 'No.46'
    batch_size = 2
    star_epoch = 0
    end_epoch = 50
    lr = 1e-3 # the initial lr is 1e-4, and multiply 0.8 for every 10k iterations.

    checkpoint_filepath = './checkpoint'
    weights_filepath = './weights'

    test_images_folder = './test/images'
    logs_dir = os.path.join('./test/tensorboard-logs', num_folder)
    result_folder = os.path.join('./test/result', num_folder)

    shutil.rmtree(logs_dir, ignore_errors=True)
    shutil.rmtree(result_folder, ignore_errors=True)

    os.makedirs(checkpoint_filepath, exist_ok=True)
    os.makedirs(weights_filepath, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


    #net = CRAFT()
    net = CRAFT(pretrained=True, freeze=False)
    # 首先使用合成数据集（SynthText）训练50K个迭代
    # 第二个模型同时在IC13和IC17上进行训练，用于评估其他五个数据集。没有额外的图像用于训练。 微调的迭代次数设置为25k
    #net.load_state_dict(copyStateDict(torch.load(os.path.join(weights_filepath, 'Syndata+IC13+IC17.pth'))))
    #net.load_state_dict(craft_dict_adapt(torch.load(os.path.join(weights_filepath, 'syn800k.pth'))))
    #net.load_state_dict(torch.load(os.path.join(weights_filepath, 'Syndata.pth')))


    #train_dataset = SynthText.SynthText(image_size=image_size,
    #                    data_dir_path="E:/datasets/SynthText/SynthText")
    total = train_dataset.__len__() // batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False)

    net = net.cuda()

    # for name, param in net.named_parameters():
    #     if "conv_cls" not in name:
    #         param.requires_grad = False


    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    #criterion = Maploss()
    criterion = torch.nn.MSELoss(reduce=False, size_average=False)
    net.train()
    to_img = transforms.ToPILImage()

    for epoch_index in range(star_epoch, end_epoch):
        
        loss_value = 0
        scheduler.step()
        print('scheduler lr: {}'.format(scheduler.get_last_lr()))

        for index, (images, region_gt, affinity_gt) in enumerate(train_loader):

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
            if (index + 1) % 100 == 0:
                print('epoch {}/{}\tbatch {}/{} \ttraining loss {:.8f}'.format(epoch_index +1, end_epoch, index + 1, len(train_loader), loss_value / 100))
                with SummaryWriter(log_dir=logs_dir, flush_secs=2, comment='train') as writer:
                    #writer.add_histogram('his/loss', loss_value / 100, epoch_index * 10 + index)
                    writer.add_scalar('data/loss', loss_value / 100, epoch_index * total + index + 1)
                loss_value = 0

            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth'

            # if (index + 1) % 1000 == 0:
            #     checkpoint_filename = os.path.join(checkpoint_filepath, 'checkpoint_iterate_' + repr(index + 1) + '.pth')
            #     print('Saving Check Point to {}'.format(checkpoint_filename))
            #     torch.save(net.state_dict(), checkpoint_filename)
            #     val(checkpoint_filename)

        if (epoch_index + 1) % 1 == 0:
            checkpoint = 'checkpoint_epoch_' + repr(epoch_index + 1)        
            checkpoint_filename = os.path.join(checkpoint_filepath, checkpoint + '.pth')
            print('Saving Check Point to {}'.format(checkpoint_filename))
            torch.save(net.state_dict(), checkpoint_filename)        

            text_dtc = TD(weight_path=checkpoint_filename)
            text_dtc.draw_to_file(test_images_folder, dist_dir=os.path.join(result_folder, checkpoint), heatmap=True)

            
        
            
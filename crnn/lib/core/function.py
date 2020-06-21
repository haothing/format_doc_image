from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)

        # inference
        preds = model(inp).cpu()

        # compute loss
        batch_size = inp.size(0)
        text, length = converter.encode(labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'batch: {0}/{1}\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:e}\t'.format(i + 1, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()

    return losses.avg

def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct, total = 0, 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                for s_i in range(len(target)):
                    try:
                        pred_s = pred[s_i]
                    except IndexError:
                        continue

                    if pred_s == target[s_i]:
                        n_correct += 1

            total += len(target) * config.TEST.BATCH_SIZE_PER_GPU
            
            #if (i + 1) % config.PRINT_FREQ == 0:
            #    print('Epoch: [{0}][{1}/{2}]'.format(epoch, i + 1, len(val_loader)))

            if i == config.TEST.NUM_TEST:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('{:80}        [ {}   <>   {} ]'.format(raw_pred, pred, gt))

    accuracy = n_correct / total
    print('correct: {}/{} \ttest loss: {:.4f} \taccuray: {:.4f}'.format(n_correct, total, losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy
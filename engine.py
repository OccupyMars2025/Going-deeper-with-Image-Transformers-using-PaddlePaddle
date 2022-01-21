# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
# import math
# import sys
from typing import Iterable, Optional
import logging
from pathlib import Path

# import torch
import paddle

from ppimm.data import Mixup
from ppimm.utils import accuracy #, ModelEma

# from losses import DistillationLoss
import utils


# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True):
def train_one_epoch(model, criterion, data_loader, optimizer,
                    epoch, mixup_fn: Optional[Mixup] = None, print_freq=10, output_dir: str=''):
    # model.train(set_training_mode)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10

    batch_id = 0
    output_dir = Path(output_dir)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)
        original_targets = targets
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = criterion(samples, outputs, targets)
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if batch_id % 10 == 0 and utils.is_main_process():
            acc1 = paddle.metric.accuracy(outputs, paddle.squeeze(original_targets, axis=-1), k=1)
            print("epoch: {}, batch_id: {}, loss is: {}, acc1 is: {}".format(epoch, batch_id, loss.numpy(), acc1.numpy()))
            logging.info("epoch: {}, batch_id: {}, loss is: {}, acc1 is: {}".format(epoch, batch_id, loss.numpy(), acc1.numpy()))

        if batch_id % 10 == 0 and utils.is_main_process():
            acc1 = paddle.metric.accuracy(outputs, paddle.squeeze(original_targets, axis=-1), k=1)
            checkpoint_path = output_dir / 'checkpoint_epoch_{}_batchid_{}_acc1_{}.pdparams'.format(epoch, batch_id, acc1.item())
            paddle.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, str(checkpoint_path))

        batch_id += 1

        # loss_value = loss.item()
        #
        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        # optimizer.zero_grad()

        # # this attribute is added by ppimm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=is_second_order)

        # torch.cuda.synchronize()
        # if model_ema is not None:
        #     model_ema.update(model)

        metric_logger.update(loss=loss.item())
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(data_loader, model, device):
@paddle.no_grad()
def evaluate(data_loader, model):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, print_freq=10, header=header):
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        # # compute output
        # with torch.cuda.amp.autocast():
        #     output = model(images)
        #     loss = criterion(output, target)
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import math
import sys
from typing import Iterable

import torch

from tools import misc, lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    amp=True, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device, non_blocking=True) for x in inputs]
        else:
            inputs = inputs.to(device, non_blocking=True)
        if isinstance(targets, list):
            targets = torch.cat(targets, -1).to(device, non_blocking=True)
        else:
            targets = targets.to(device, non_blocking=True)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    loss = sum([criterion(output, targets) for output in outputs])
                else:
                    loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                loss = sum([criterion(output, targets) for output in outputs])
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for inputs, targets in metric_logger.log_every(data_loader, 10, header):
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device, non_blocking=True) for x in inputs]
        else:
            inputs = inputs.to(device, non_blocking=True)
        if isinstance(targets, list):
            targets = torch.cat(targets, -1).to(device, non_blocking=True)
        else:
            targets = targets.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        mse = torch.nn.functional.mse_loss(outputs, targets)
        mae = torch.nn.functional.l1_loss(outputs, targets)

        batch_size = targets.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae'].update(mae.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MAE {mae.global_avg:.3f} MSE {mse.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mae=metric_logger.mae, mse=metric_logger.mse, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

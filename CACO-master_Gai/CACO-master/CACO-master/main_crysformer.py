import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from tools.weight_decay import add_weight_decay
import tools.misc as misc
from tools.misc import NativeScalerWithGradNormCount as NativeScaler

# from models.alignn import ALIGNN
from models.crysformer import Crysformer
from loader.loader import get_dataset
from loader.transforms import *

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('ALIGNN fine-tuning for property regression', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='crysformer_v10', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--inputs', default=['graph', 'line_graph'], type=str, nargs='+',
                        help='List of input features')
    parser.add_argument('--targets', default=['e_form'], type=str, nargs='+',
                        help='List of target propties')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='convert bn to sync_bn')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Finetuning params
    parser.add_argument('--pretrain', default='',
                        help='Load pertrain from checkpoint')

    # Dataset parameters
    parser.add_argument('--dataset', default='megnet', type=str,
                        help='dataset')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--ratio_train_val_test', default=None, type=float, nargs='+',
                        help='train val test dataset split ratio')
    parser.add_argument('--n_train_val_test', default=[60000,5000,4239], type=int, nargs='+',
                        help='train val test dataset split num')
    
    # Training parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='checkpoint saving interval')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if 'line_graph' in args.inputs:
        transform = Compose([
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollecting(args.inputs, args.targets)
        ])

    else:
        transform = Compose([
            AddNodeFeature(),
            AddEdgeFeature(),
            GraphCollecting(args.inputs, args.targets)
        ])
    train_dataset, val_dataset, test_dataset = \
        get_dataset(args.dataset, args.data_path, args.ratio_train_val_test, args.n_train_val_test, transform, True, line_graph='line_graph' in args.inputs)

    # if args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     train_sampler = torch.utils.data.DistributedSampler(
    #         train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(train_sampler))
    #     if args.dist_eval:
    #         if len(val_dataset) % num_tasks != 0:
    #             print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
    #                   'This will slightly alter validation results as extra duplicate entries are added to achieve '
    #                   'equal num of samples per-process.')
    #         val_sampler = torch.utils.data.DistributedSampler(
    #             val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    #         test_sampler = torch.utils.data.DistributedSampler(
    #             test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    #     else:
    #         val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    #         test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #     val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    # if global_rank == 0 and args.log_dir is not None and not args.eval:
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        logfile = 'log_' + time.strftime('%Y%m%d_%H%M',time.localtime()) + '.txt'
        with open(os.path.join(args.output_dir, logfile), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(vars(args)) + "\n")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=train_dataset.collect(),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=val_dataset.collect(),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=test_dataset.collect(),
    )

    # model = ALIGNN()
    model = Crysformer(inputs=args.inputs, targets=args.targets)

    if args.pretrain and not args.eval:
        checkpoint = torch.load(args.pretrain, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.pretrain)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    # if args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(val_dataloader, model, device)
        print(f"MAE of the network on the {len(val_dataset)} test crystals: {test_stats['mae']:.3f}")
        print(f"MSE of the network on the {len(val_dataset)} test crystals: {test_stats['mse']:.3f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_mae = 10.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, train_dataloader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            amp=False,
            args=args
        )
        if args.output_dir and (epoch % args.save_interval == 0 or epoch == args.epochs-1):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(val_dataloader, model, device, amp=False)
        print(f"MAE of the network on the {len(val_dataset)} validation crystals: {val_stats['mae']:.3f}")
        print(f"MSE of the network on the {len(val_dataset)} validation crystals: {val_stats['mse']:.3f}")
        min_mae = min(min_mae, val_stats["mae"])
        print(f'Min MAE: {min_mae:.3f}')

        if log_writer is not None:
            log_writer.add_scalar('perf/val_mae', val_stats['mae'], epoch)
            log_writer.add_scalar('perf/val_mse', val_stats['mse'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, logfile), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    test_stats = evaluate(test_dataloader, model, device, amp=False)
    print(f"MAE of the network on the {len(test_dataset)} test crystals: {test_stats['mae']:.3f}")
    print(f"MSE of the network on the {len(test_dataset)} test crystals: {test_stats['mse']:.3f}")
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, logfile), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

import os
import argparse
import glob
import sys
import time
import traceback
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from model import AIMnet
from utils import *
from trainer import Trainer


RUN_REPORT = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def find_latest_checkpoint(save_path):
    latest = os.path.join(save_path, 'latest.pth')
    if os.path.isfile(latest):
        return latest

    checkpoints = glob.glob(os.path.join(save_path, 'model_e*.pth'))
    if not checkpoints:
        return None

    def epoch_from_name(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.split('model_e')[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=epoch_from_name)


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def write_run_report(args, trainer, status, start_time, error=None):
    report_dir = os.path.join(args.save_path, 'reports')
    os.makedirs(report_dir, exist_ok=True)

    end_time = time.time()
    start_dt = datetime.fromtimestamp(start_time)
    end_dt = datetime.fromtimestamp(end_time)
    last_epoch = trainer.summary.get('last_epoch', 0) if trainer is not None else 0
    timestamp = start_dt.strftime('%Y%m%d_%H%M%S')
    report_name = f'train_{timestamp}_{status}_epoch{last_epoch}.md'
    report_path = os.path.join(report_dir, report_name)

    history = trainer.summary.get('history', []) if trainer is not None else []
    best = trainer.summary.get('best', {}) if trainer is not None else {}
    last = history[-1] if history else {}

    lines = [
        '# Training Run Report',
        '',
        f'- Status: `{status}`',
        f'- Start time: `{start_dt.strftime("%Y-%m-%d %H:%M:%S")}`',
        f'- End time: `{end_dt.strftime("%Y-%m-%d %H:%M:%S")}`',
        f'- Duration: `{format_duration(end_time - start_time)}`',
        f'- Working directory: `{os.getcwd()}`',
        f'- Command: `{" ".join(sys.argv)}`',
        '',
        '## Arguments',
        '',
        '```text',
    ]
    for key, value in sorted(vars(args).items()):
        lines.append(f'{key}: {value}')
    lines.extend([
        '```',
        '',
        '## Result',
        '',
        f'- Last epoch: `{last_epoch}`',
        f'- Last main loss: `{last.get("main_loss", "N/A")}`',
        f'- Last train PSNR: `{last.get("train_psnr", "N/A")}`',
        f'- Last val PSNR: `{last.get("val_psnr", "N/A")}`',
        f'- Last val SSIM: `{last.get("val_ssim", "N/A")}`',
        f'- Best val PSNR: `{best.get("val_psnr", "N/A")}` at epoch `{best.get("epoch", "N/A")}`',
        f'- Best val SSIM: `{best.get("val_ssim", "N/A")}` at epoch `{best.get("ssim_epoch", "N/A")}`',
        f'- Latest checkpoint: `{os.path.join(args.save_path, "latest.pth")}`',
        '',
        '## Epoch History',
        '',
        '| Epoch | Main Loss | Train PSNR | Val PSNR | Val SSIM | LR |',
        '|---:|---:|---:|---:|---:|---:|',
    ])
    for row in history:
        lines.append(
            f'| {row["epoch"]} | {row["main_loss"]:.6f} | {row["train_psnr"]:.6f} | '
            f'{row["val_psnr"]:.6f} | {row["val_ssim"]:.6f} | {row["lr"]:.8f} |'
        )

    if error is not None:
        lines.extend([
            '',
            '## Error',
            '',
            '```text',
            ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
            '```',
        ])

    with open(report_path, 'w') as report_file:
        report_file.write('\n'.join(lines) + '\n')

    print('')
    print('========== Training Run Summary ==========')
    print(f'Status: {status}')
    print(f'Command: {" ".join(sys.argv)}')
    print(f'Duration: {format_duration(end_time - start_time)}')
    print(f'Last epoch: {last_epoch}')
    print(f'Last main loss: {last.get("main_loss", "N/A")}')
    print(f'Last train PSNR: {last.get("train_psnr", "N/A")}')
    print(f'Last val PSNR: {last.get("val_psnr", "N/A")}')
    print(f'Last val SSIM: {last.get("val_ssim", "N/A")}')
    print(f'Best val PSNR: {best.get("val_psnr", "N/A")} at epoch {best.get("epoch", "N/A")}')
    print(f'Training report saved to: {report_path}')
    print('==========================================')
    return report_path


def main(gpu, args):
    global RUN_REPORT
    args.local_rank = gpu
    # random seed
    setup_seed(2022)
    # load data
    train_folder = args.data_dir
    paired_dataset = TrainLabeled(dataroot=train_folder, phase='labeled1', finesize=args.crop_size)
    unpaired_dataset = TrainUnlabeled(dataroot=train_folder, phase='unlabeled', finesize=args.crop_size)
    val_dataset = ValLabeled(dataroot=train_folder, phase='val', finesize=args.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(paired_dataset, batch_size=args.train_batchsize, sampler=paired_sampler)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=args.train_batchsize, sampler=unpaired_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, sampler=val_sampler)
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))
    # create model
    net = AIMnet()
    ema_net = AIMnet()
    ema_net = create_emamodel(ema_net)
    print('student model params: %d' % count_parameters(net))
    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = None
    start_time = time.time()
    status = 'completed'
    error = None
    try:
        trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                          unsupervised_loader=unpaired_loader,
                          val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)
        trainer.train()
    except KeyboardInterrupt as exc:
        status = 'interrupted'
        error = exc
        raise
    except Exception as exc:
        status = 'failed'
        error = exc
        raise
    finally:
        writer.close()
        RUN_REPORT = write_run_report(args, trainer, status, start_time, error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-g', '--gpus', default=2, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--train_batchsize', default=8, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=4, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume from checkpoint')
    parser.add_argument('--resume_path', default='', type=str, help='checkpoint path for resume')
    parser.add_argument('--auto_resume', default=False, type=str2bool, help='resume from latest checkpoint in save_path')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='/path/to/pretained/net.pth', type=str, help='if pretrained')
    parser.add_argument('--data_dir', default='./data', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--save_period', default=20, type=int, help='save numbered checkpoint every N epochs')
    parser.add_argument('--log_dir', default='./model/log', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if args.auto_resume and not args.resume_path:
        latest_checkpoint = find_latest_checkpoint(args.save_path)
        if latest_checkpoint is None:
            print(f'No checkpoint found in {args.save_path}; starting from scratch.')
        else:
            args.resume = True
            args.resume_path = latest_checkpoint
            print(f'Auto-resume checkpoint: {args.resume_path}')
    main(-1, args)

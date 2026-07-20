import os
import argparse
import glob
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from model import AIMnet
from utils import *
from trainer import Trainer


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


def main(gpu, args):
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
    trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)

    trainer.train()
    writer.close()


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

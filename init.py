import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import model_HorizonNet
from dataset import FlatLayoutDataset


def init(args):
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    dataset_train = FlatLayoutDataset(args.imgroot, args.gtpath,
                                  hw=(args.main_h, args.main_w),
                                  flip=args.flip, gamma=args.gamma,
                                  outy_mode=args.outy_mode, outy_val=(args.outy_val_up, args.outy_val_bt),
                                  y_step=args.y_step, gen_doncare=args.ori_res_loss)
    loader_train = DataLoader(dataset_train, args.batch_size_train,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())
    if args.valid_imgroot:
        dataset_valid = FlatLayoutDataset(args.valid_imgroot, args.valid_gtpath,
                                      hw=(args.main_h, args.main_w),
                                      flip=False, gamma=False,
                                      outy_mode=args.outy_mode, outy_val=(args.outy_val_up, args.outy_val_bt),
                                      y_step=args.y_step, gen_doncare=args.ori_res_loss)
        loader_valid = DataLoader(dataset_valid, args.batch_size_valid,
                                  shuffle=False, drop_last=False,
                                  num_workers=args.num_workers,
                                  pin_memory=not args.no_cuda)
    else:
        loader_valid = None

    # Create model
    if args.net in ['HorizonNet', 'LowResHorizonNet']:
        Model = getattr(model_HorizonNet, args.net)
    else:
        Model = getattr(model, args.net)
    model_kwargs = {'init_bias': args.init_bias}
    if args.backbone:
        model_kwargs['backbone'] = args.backbone
    if args.dilate_scale:
        model_kwargs['dilate_scale'] = args.dilate_scale
    if args.pred_cor:
        model_kwargs['pred_cor'] = args.pred_cor
    net = Model(**model_kwargs).to(device)
    assert -1 <= args.freeze_earlier_blocks and args.freeze_earlier_blocks <= 4
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            print('Freeze block%d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    # Create optimizer
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    args.max_iters = args.epochs * len(loader_train)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.cur_iter = 0
    optimizer = getattr(optim, args.optimizer)(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create tensorboard for monitoring training
    tb_path = os.path.join(args.logs, args.id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    return device, loader_train, loader_valid, net, model_kwargs, optimizer, tb_writer

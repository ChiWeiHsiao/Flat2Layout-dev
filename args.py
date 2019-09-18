import os
import sys
import time
import argparse
import configparser
from threading import Thread

def input_countdown(timeout=30):
    thread = Thread(target=input)
    thread.daemon = True
    thread.start()
    start_time = time.time()
    while time.time() - start_time < timeout:
        left_time = int(timeout - (time.time() - start_time))
        print('Press any key to continue ... [%3d]' % left_time, end='\r')
        if not thread.isAlive():
            break
        time.sleep(0.1)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config')

    # Read exp setting from config file first
    config_args, remaining_argv = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(config_args.config)
    default = dict(config['EXP_SETTING'])
    default['init_bias'] = list(map(float, default.get('init_bias', '').split(',')))

    # Placeholder setting from config
    parser.add_argument('--id', help='experiment id to name checkpoints and logs')
    parser.add_argument('--imgroot')
    parser.add_argument('--gtpath')
    parser.add_argument('--valid_imgroot')
    parser.add_argument('--valid_gtpath')
    # Loss related
    parser.add_argument('--loss', choices=['l1', 'l2', 'huber', 'berhu'])
    parser.add_argument('--huber_const', default=0.2, type=float, help='constant in huber loss')
    parser.add_argument('--lap_order', default=0, type=int, help='gradient form by pred. and gt.')
    parser.add_argument('--guide_gain', default=0, type=int)
    parser.add_argument('--y_step', default=1, type=int, help='resample y from [B,C,W] to [B,C,W/y_step]')
    parser.add_argument('--ori_res_loss', default=1, type=int, help='output low resolution, compute loss at full res')
    parser.add_argument('--weight_cor', default=1, type=float, help='loss=weight_cor*loss_cor+loss_boundary')
    parser.add_argument('--weight_key', default=1, type=float, help='loss=weight_key*loss_key+loss_boundary')
    parser.add_argument('--pos_weight_cor', default=5, type=float, help='loss=weight_cor*loss_cor+loss_boundary')
    parser.add_argument('--no_reg_loss', default=0, type=int)
    parser.add_argument('--ori_res_cor', default=0, type=int)
    parser.add_argument('--bon_sample_rates', default=0, type=int, help='if r>0, sample bondary w/ rate [1,...,r]')
    # Upsample related
    parser.add_argument('--use_dontcare', type=int)
    parser.add_argument('--upsample_lr_pad', type=int)
    parser.add_argument('--gt_down_upsample', type=int)
    # Model related
    parser.add_argument('--net')
    parser.add_argument('--gray_mode', type=int)
    parser.add_argument('--use_rnn', default=1, type=int)
    parser.add_argument('--branches', type=int)
    parser.add_argument('--bn_momentum', type=float)
    parser.add_argument('--backbone', default='resnext50_32x4d')
    parser.add_argument('--dilate_scale', type=int)
    parser.add_argument('--init_bias', nargs='*', type=float)
    parser.add_argument('--freeze_earlier_blocks', type=int)
    parser.add_argument('--freeze_bn', type=int)
    parser.add_argument('--pred_cor', type=int, help='let model predict corner or not')
    parser.add_argument('--pred_key', type=int, help='let model predict wall-wall keypoint or not')
    parser.add_argument('--drop_p', default=0.5, type=float)
    parser.add_argument('--freeze_stage1', default=0, type=int, help='only works for TwoStageNet')
    parser.add_argument('--c_out_bon', type=int, help='specify if want to load pretrained y_reg w/ inconsistent #out channel')
    # Dataset related
    parser.add_argument('--resize_h', type=int)
    parser.add_argument('--flip', type=int, help='use flip augmentation')
    parser.add_argument('--gamma', type=int, help='use gamma augmentation')
    parser.add_argument('--outy_mode', choices=['linear', 'constant'], help='setting value of y when it is outside image plane')
    parser.add_argument('--outy_val_up', default=-1.05, type=float, help='setting value of y when it is outside image plane')
    parser.add_argument('--outy_val_bt', default=1.05, type=float, help='setting value of y when it is outside image plane')
    parser.add_argument('--cor_mode', choices=['binary', 'exp'], help='exp to broadcast cor/key GT')
    parser.add_argument('--main_h', type=int)
    parser.add_argument('--main_w', type=int)
    parser.add_argument('--scales', nargs='*', type=float)
    parser.add_argument('--num_workers', type=int, help='numbers of workers for dataloaders')
    parser.add_argument('--batch_size_train', type=int, help='training mini-batch size')
    parser.add_argument('--batch_size_valid', type=int, help='validation mini-batch size')
    # Training related
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float, help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float, help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='numbers of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, help='epochs to train')
    parser.add_argument('--septrain', default=0, type=int)
    parser.add_argument('--finetune_cor', default=0, type=int)
    parser.add_argument('--load_pretrain', type=str, default=None)
    parser.add_argument('--epochs_bon', type=float, help='epochs to train only boundary y_reg')
    # Misc
    parser.add_argument('--no_cuda', help='disable cuda', type=int)
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--save_every', type=int, help='epochs frequency to save state_dict')
    parser.add_argument('--ckpt', help='folder to output checkpoints')
    parser.add_argument('--logs', help='folder to logging')
    parser.set_defaults(**default)

    # Read from remaining command line setting (replace setting in config)
    args = parser.parse_args(remaining_argv)

    # Ask human check again
    print(args.__dict__)
    print()
    print('Please check again the setting. (especially id)')
    input_countdown(timeout=60)
    print()
    print('GOGO')

    return args

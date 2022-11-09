import argparse

from numpy import true_divide
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from yacs.config import CfgNode as CN
import copy

import datasets.ssdg_pacs
import datasets.ssdg_officehome

import trainers.adg


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    cfg.TRAINER.ADG.UPDATE_DATA = (args.update_data>0)


def extend_cfg(cfg):
    cfg.TRAINER.UPDATE_EPOCHS = []
    cfg.TRAINER.ADG = CN()
    cfg.TRAINER.ADG.STRONG_TRANSFORMS = ()
    cfg.TRAINER.ADG.CC_OPTIM = copy.deepcopy(cfg.OPTIM) 
    cfg.TRAINER.ADG.DC_OPTIM = copy.deepcopy(cfg.OPTIM) 

    cfg.TRAINER.ADG.SAVE_EPOCH = 0
    cfg.TRAINER.ADG.CONSISTENCY = True
    cfg.TRAINER.ADG.CLUSTER_P = 30
    cfg.TRAINER.ADG.CLUSTER_INIT = 50
    cfg.TRAINER.ADG.CLUSTER_STATIC = True
    cfg.TRAINER.ADG.CLUSTER_EPOCH = 2
    cfg.TRAINER.ADG.LARGER_R = 1.5
    cfg.TRAINER.ADG.ALPHA = 0.3
    # cfg.TRAINER.ADG.BETA = 0.5
    cfg.TRAINER.ADG.BETA = []

    cfg.TRAINER.ADG.ALMETHOD = 'ours3'
    cfg.TRAINER.ADG.USEDOMAIN = True
    cfg.TRAINER.ADG.DIVERSITY = True
    cfg.TRAINER.ADG.GAMMA = [0.33,0.33,0.34]
    cfg.TRAINER.ADG.UNCERTAINTY = 'bvsb'
    cfg.TRAINER.ADG.DOMAINESS = 'bvsb'
    cfg.TRAINER.ADG.DOMAINESS_FLIP = True

    cfg.TRAIN.DATA_MODE = ''
    cfg.DATALOADER.RETURN_IMG0 = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='positive seed value'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domain'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )

    parser.add_argument(
        '--update-data',
        type=int,
        default=1,
        help='whether update the labeled data after each epoch, val > 0: True; val <= 0: False'
    )

    args = parser.parse_args()

    main(args)

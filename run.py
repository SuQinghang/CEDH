import argparse
import random

import numpy as np
import torch
from loguru import logger

import adsh
import cedh
from data.data_loader import load_data


def run():
    args = load_config()
    save_name = '{}_{}bits_topk@{}'.format(
        args.dataset,
        args.code_length,
        args.topk,
    )
    logger.add('logs/{}/{}/{}.log'.format(args.method, args.dataset, save_name), rotation='500 MB', level='INFO')
    a = vars(args)
    logger.info('-------------------------Current Settings-------------------------')
    for key, value in a.items():
        logger.info('{} = {}'.format(key, value))
    
    torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        setup_seed(args.seed)
    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.batch_size,
        args.num_workers,
    )
    logger.info('Train Size = {}'.format(len(train_dataloader.dataset)))
    logger.info('Query Size = {}'.format(len(query_dataloader.dataset)))
    logger.info('Database Size = {}'.format(len(retrieval_dataloader.dataset)))

    if args.method == 'adsh':
        adsh.train(
            query_dataloader,
            train_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.gamma,
            args.topk,
            args.eval_epoch,
            save_name
        )
    else:
        assert args.code_length>args.original_length, 'target_length must greater than original_length!'
        cedh.train(
            dataset              = args.dataset,
            dataset_root         = args.root,
            query_dataloader     = query_dataloader,
            train_dataloader     = train_dataloader,
            retrieval_dataloader = retrieval_dataloader,

            original_method   = args.original_method,
            original_code_dir = args.original_dir,
            original_length   = args.original_length,
            target_length     = args.code_length,

            max_iter    = args.max_iter,
            max_epoch   = args.max_epoch,
            batch_size  = args.batch_size,
            num_samples = args.num_samples,
            lr          = args.lr,
            W_lambda    = args.W_lambda,
            gamma       = args.gamma,
            topk        = args.topk,
            device      = args.device,
            sim_S       = args.sim_S,
            alpha       = args.alpha,
            eval_epoch  = args.eval_epoch,
            save_name   = save_name
        )

        # logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enalbed = False

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='CEDH_PyTorch')
    parser.add_argument('--method', default='adsh', type=str,
                        help='Method name.(default:adsh)')
    parser.add_argument('--dataset', default=None, type=str,
                        help='Dataset name.')
    parser.add_argument('--root', default=None, type=str,
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate.(default: 3e-5)')
    parser.add_argument('--code-length', default=20, type=int,
                        help='Binary hash code length.(default: 20)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of loading data threads.(default: 8)')
    parser.add_argument('--topk', default=None, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=2, type=int,
                        help='Using gpu.(default: 2)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--eval-epoch', default=5, type=int,
                        help='Hyper-parameter.(default: 5)')
    parser.add_argument('--seed', default=None, type=int,
                        help='Hyper-parameter.(default:  2333)')                    

    # following parameters are required if method is set to cedh
    parser.add_argument('--original-method', default='adsh', type=str,
                        help='Original method name.')
    parser.add_argument('--original-dir', default=None, type=str,
                        help='Path of original code.')
    parser.add_argument('--original-length', default=20, type=int,
                        help='Orignal length of hash code.(default: 20)')
    parser.add_argument('--W-lambda', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--alpha', default=1, type=float,
                        help='Hyper-parameter.(default:1)')
    # set smooth for cifar-10 and imagenet and cosine for nus-wide-tc21
    parser.add_argument('--sim_S', default='smooth', type=str,
                        help='Type of similarity matrix.')


    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()

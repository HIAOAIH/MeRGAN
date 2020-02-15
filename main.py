# ACGAN references
# https://github.com/eriklindernoren/PyTorch-GAN
# https://github.com/znxlwm/pytorch-generative-model-collections
# https://github.com/znxlwm/pytorch-generative-model-collections

import argparse
import os
import MeRGAN


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to train and generate')
    parser.add_argument('--class_num', type=int, default=10, help='the number of classes you want to train')
    parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')
    parser.add_argument('--num_generated', type=int, default=4096, help='the number of images to generate in each class')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--method', type=str, choices=['jrt', 'joint_retraining', 'ra', 'replay_alignment'])

    return check_args(parser.parse_args())


def check_args(args):
    if args.method == 'jrt':
        args.method = 'joint_retraining'
    elif args.method == 'ra':
        args.method = 'replay_alignment'

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.result_dir + '/' + args.dataset):
        os.makedirs(args.result_dir + '/' + args.dataset)

    if not os.path.exists(args.result_dir + '/' + args.dataset + '/' + args.method):
        os.makedirs(args.result_dir + '/' + args.dataset + '/' + args.method)

    for i in range(args.class_num):
        if not os.path.exists(args.result_dir + '/' + args.dataset + '/' + args.method + '/to_' + str(i)):
            os.makedirs(args.result_dir + '/' + args.dataset + '/' + args.method + '/to_' + str(i))

    return args


def main():
    args = args_parser()
    if args is None:
        exit()

    mergan = MeRGAN(args)


if __name__ == '__main__':
    main()
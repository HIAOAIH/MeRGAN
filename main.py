# ACGAN references
# https://github.com/eriklindernoren/PyTorch-GAN
# https://github.com/znxlwm/pytorch-generative-model-collections
# https://github.com/znxlwm/pytorch-generative-model-collections

import argparse
import os
import copy
import torch
from MeRGAN import MeRGAN
from TestGenerator import TestGenerator
#from apex import amp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNSIT', 'SVHN'], help='dataset to train and generate')
    parser.add_argument('--class_num', type=int, default=10, help='the number of classes you want to train')
    parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')
    parser.add_argument('--num_generated', type=int, default=4096, help='the number of images to generate in each class')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--network_dir', type=str, default='network')
    parser.add_argument('--method', type=str, default='joint_retraining', choices=['jrt', 'joint_retraining', 'ra', 'replay_alignment'])
    parser.add_argument('--work', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--task', type=str, default='to_9', choices=['to_9', 'to_4'], help='the number of classes to classify')

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

    if not os.path.exists(args.network_dir):
        os.makedirs(args.network_dir)

    if not os.path.exists(args.network_dir + '/' + args.dataset):
        os.makedirs(args.network_dir + '/' + args.dataset)

    if not os.path.exists(args.network_dir + '/' + args.dataset + '/' + args.method):
        os.makedirs(args.network_dir + '/' + args.dataset + '/' + args.method)

    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    if args.work == 'train':
        mergan = MeRGAN(args)
        #dummy_opt = torch.optim.Adam(mergan.parameters(), lr=1e-3)
        #mergan, optimizer = amp.initialize(mergan, dummy_opt)
        if args.method == 'joint_retraining':
            for i in range(10):
                if i == 0:
                    mergan.init_ACGAN(mergan.data_list[i], i)
                else:
                    G = mergan.ACGAN.G
                    D = mergan.ACGAN.D
                    mergan.init_ACGAN(generated_data, i, G, D)
                generated_data = mergan.generate_trainset()
                if i < 9:
                    generated_data.concat_datasets(mergan.data_list[i + 1])
                if i == 4:
                    torch.save(mergan.ACGAN.G.state_dict(), './network/' + args.dataset + '/' + args.method + '/generator_' + args.method + '_to_4.pt')
            torch.save(mergan.ACGAN.G.state_dict(), './network/' + args.dataset + '/' + args.method + '/generator_' + args.method + '_to_9.pt')
        else:
            for i in range(10):
                if i == 0:
                    mergan.init_ACGAN(mergan.data_list[i], i)
                else:
                    mergan.init_ACGAN(mergan.data_list[i], i, G_past, D_past)
                G_past = copy.deepcopy(mergan.ACGAN.G)
                D_past = copy.deepcopy(mergan.ACGAN.D)
                if i == 4:
                    torch.save(mergan.ACGAN.G.state_dict(), './network/' + args.dataset + '/' + args.method + '/generator_' + args.method + '_to_4.pt')
            torch.save(mergan.ACGAN.G.state_dict(), './network/' + args.dataset + '/' + args.method + '/generator_' + args.method + '_to_9.pt')

    elif args.work == 'test':
        test_generator = TestGenerator(args)
        test_generator.test()


if __name__ == '__main__':
    main()

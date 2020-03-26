import torch
import copy
from ACGAN import ACGAN
from CustomDataset import CustomDataset
from torchvision import datasets, transforms


class MeRGAN(object):
    def __init__(self, args):
        # dataset == MNIST 면 input_dim=100, total_class_num=10
        # ACGAN에 result_dir 전달할 때는 result_dir + '/' + dataset으로
        self.dataset_name = args.dataset
        self.total_class_num = args.class_num
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.num_in_class = args.num_generated
        self.method = args.method
        self.result_dir = args.result_dir + '/' + args.dataset + '/' + args.method
        self.class_array = []
        self.data_list = []
        if self.dataset_name == 'MNIST':
            d = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(28, 28), transforms.ToTensor(),
                                                             transforms.Normalize((0.5, ), (0.5, ))]))
            self.image_size = 28
        elif self.dataset_name == 'SVHN':
            d = datasets.SVHN('./data', download=True, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
            self.image_size = 32
        self.dataset_with_class(d)

    def dataset_with_class(self, dataset):
        # It works MNIST only
        if self.dataset_name == 'MNIST':
            for i in range(self.total_class_num):
                temp = copy.deepcopy(dataset)
                idx = dataset.targets == i
                temp.targets = dataset.targets[idx]
                temp.data = dataset.data[idx]
                self.data_list.append(temp)
        elif self.dataset_name == 'SVHN':
            for i in range(self.total_class_num):
                temp = copy.deepcopy(dataset)
                idx = dataset.labels == i
                temp.targets = dataset.labels[idx]
                temp.data = dataset.data[idx]
                self.data_list.append(temp)

    def init_ACGAN(self, dataset, class_index, G_past=None, D_past=None):
        if class_index in self.class_array:
            raise Exception("this class is already trained.")
        self.ACGAN = ACGAN(data_loader=torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True),
                           dataset=self.dataset_name, class_index=len(self.class_array) + 1, method=self.method,
                           result_dir=self.result_dir, batch_size=self.batch_size, epoch=self.epoch)
        self.ACGAN.train(G_past, D_past)
        self.class_array.append(class_index)

    def generate_trainset(self):
        # try:
        #     if self.class_idx != class_idx:
        #         raise Exception("ClassIndexMissMatchError")
        # except AttributeError:
        #     raise Exception("ACGAN in MeRGAN does not init yet. Init that first")
        self.ACGAN.G.eval()
        n = len(self.class_array)
        for i in range((self.num_in_class * n // self.batch_size)):
            y_mer = torch.LongTensor(self.batch_size, 1).random_(0, n)
            y_mer_one_hot = torch.zeros(self.batch_size, self.total_class_num).scatter_(1, y_mer.type(torch.LongTensor), 1)
            z_mer = torch.rand(self.batch_size, 118)

            if self.ACGAN.gpu_mode:
                y_mer, y_mer_one_hot, z_mer = y_mer.cuda(), y_mer_one_hot.cuda(), z_mer.cuda()

            x_mer = self.ACGAN.G(z_mer, y_mer_one_hot)

            if self.ACGAN.gpu_mode:
                x_mer, y_mer = x_mer.cpu().detach().view(-1, self.image_size, self.image_size), y_mer.cpu().detach()
            else:
                x_mer, y_mer = x_mer.detach().view(-1, self.image_size, self.image_size), y_mer.detach()

            if i == 0:
                data_list = CustomDataset(x_mer, y_mer.view(-1), self.dataset_name)
            else:
                data_list.append(x_mer, y_mer.view(-1))

            # print("{}% of train data generated".format(100. * ((i + 1) * self.batch_size) / self.num_in_class * n))

        return data_list
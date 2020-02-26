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
        if args.dataset == 'MNIST':
            d = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([transforms.Resize(28, 28), transforms.ToTensor(),
                                                             transforms.Normalize((0.5, ), (0.5, ))]))
        self.dataset_with_class(d)

    def dataset_with_class(self, dataset):
        # It works MNIST only
        for i in range(self.total_class_num):
            temp = copy.deepcopy(dataset)
            idx = dataset.targets == i
            temp.targets = dataset.targets[idx]
            temp.data = dataset.data[idx]
            self.data_list.append(temp)

    def init_ACGAN(self, dataset, class_index, G_past=None):
        if class_index in self.class_array:
            raise Exception("this class is already trained.")
        self.ACGAN = ACGAN(data_loader=torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True),
                           class_index=len(self.class_array) + 1, method=self.method, result_dir=self.result_dir,
                           batch_size=self.batch_size, epoch=self.epoch)
        self.ACGAN.train(G_past)
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
            z_mer = torch.rand(self.batch_size, 100)

            if self.ACGAN.gpu_mode:
                y_mer, y_mer_one_hot, z_mer = y_mer.cuda(), y_mer_one_hot.cuda(), z_mer.cuda()

            x_mer = self.ACGAN.G(z_mer, y_mer_one_hot)

            if self.ACGAN.gpu_mode:
                x_mer, y_mer = x_mer.cpu().detach().view(-1, 28, 28), y_mer.cpu().detach()
            else:
                x_mer, y_mer = x_mer.detach().view(-1, 28, 28), y_mer.detach()

            if i == 0:
                data_list = CustomDataset(x_mer, y_mer.view(-1))
            else:
                data_list.append(x_mer, y_mer.view(-1))

        return data_list
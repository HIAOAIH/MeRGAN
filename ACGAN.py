import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio


class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_channel=1, input_size=28, total_class_num=10, class_num=10):
        super(Generator, self).__init__()
        # initial parameters are optimized to MNIST dataset
        self.noise_dim = noise_dim
        self.output_channel = output_channel
        self.input_size = input_size
        self.total_class_num = total_class_num
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.noise_dim + self.total_class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) ** 2),
            nn.BatchNorm1d(128 * (self.input_size // 4) ** 2),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise, label):
        # concat random noise and answer label
        x = torch.cat([noise, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, self.input_size // 4, self.input_size // 4)
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_channel=1, dc_dim=1, total_class_num=10, input_size=28):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.dc_dim = dc_dim
        self.total_class_num = total_class_num
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) ** 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num)
        )

    def forward(self, image):
        x = self.conv(image)
        x = x.view(-1, 128 * (self.input_size // 4) ** 2)
        x = self.fc(x)
        d = self.dc(x)
        c = self.cl(x)

        # d: discriminate real or fake
        # c: classify the class
        return d, c


class ACGAN(object):
    def __init__(self, data_loader, sample_num=100, noise_dim=100, total_class_num=10, class_index=10,
                 method='ra', result_dir='result', batch_size=64, lr=0.0002, beta1=0.5, beta2=0.999, gpu_mode=True, epoch=20):
        self.sample_num = sample_num
        self.noise_dim = noise_dim
        self.total_class_num = total_class_num
        self.class_index = class_index
        self.method = method
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        if gpu_mode:
            if torch.cuda.is_available():
                self.gpu_mode = True
            else:
                print("There isn't any available GPU")
                self.gpu_mode = False
        else:
            self.gpu_mode = False
        self.epoch = epoch
        self.train_history = {'D_loss': [], 'G_loss': [], 'per_epoch_time': [], 'total_time': []}

        # loss to discriminate input image real or fake
        self.BCE_loss = nn.BCELoss()
        # loss to classify specific class of input image
        self.CE_loss = nn.CrossEntropyLoss()
        if method == 'ra':
            # loss to train current Generator using past Generator
            self.method = 'ra'
            self.MSE_loss = nn.MSELoss()

        self.data_loader = data_loader
        data = self.data_loader.__iter__().__next__()[0]

        self.G = Generator(self.noise_dim, data.shape[1], data.shape[2], self.total_class_num)
        self.D = Discriminator(data.shape[1], data.shape[1], data.shape[2], self.total_class_num)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if self.gpu_mode:
            self.G, self.D = self.G.cuda(), self.D.cuda()
            self.BCE_loss, self.CE_loss = self.CE_loss.cuda()
            if self.method == 'ra':
                self.MSE_loss = self.MSE_loss.cuda()

    def train(self):
        y_real, y_fake = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            y_real, y_fake = y_real.cuda(), y_fake.cuda()

        print("start training")

        self.D.train()
        for epoch in range(self.epoch):
            self.G.train()

            for idx, (x, y) in enumerate(self.data_loader):
                if idx == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z = torch.rand(self.batch_size, self.noise_dim)

                y_vec = torch.zeros(self.batch_size, self.total_class_num).scatter_(
                    1, y.type(torch.LongTensor).unsqueeze(1), 1)
                if self.gpu_mode:
                    x, y, y_vec, z = x.cuda(), y.cuda(), y_vec.cuda(), z.cuda()


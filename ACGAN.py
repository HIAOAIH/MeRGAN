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
    def __init__(self, input_channel=1, dc_dim=1, input_size=28, total_class_num=10):
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
            nn.Linear(1024, self.dc_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.total_class_num)
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
    def __init__(self, data_loader, dataset='MNIST', sample_num=100, noise_dim=100, total_class_num=10, class_index=10,
                 method='replay_alignment', result_dir='result', batch_size=64, lr=0.0002, beta1=0.5, beta2=0.999, gpu_mode=True,
                 epoch=20):
        self.dataset = dataset
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
        if method == 'replay_alignment':
            # loss to train current Generator using past Generator
            self.method = 'replay_alignment'
            self.MSE_loss = nn.MSELoss()

        self.data_loader = data_loader

        data = self.data_loader.__iter__().__next__()[0]

        self.G = Generator(self.noise_dim, data.shape[1], data.shape[2], self.total_class_num)
        self.D = Discriminator(data.shape[1], data.shape[1], data.shape[2], self.total_class_num)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if self.gpu_mode:
            self.G, self.D = self.G.cuda(), self.D.cuda()
            self.BCE_loss, self.CE_loss = self.BCE_loss.cuda(), self.CE_loss.cuda()
            if self.method == 'replay_alignment':
                self.MSE_loss = self.MSE_loss.cuda()

    def train(self, G_past=None):
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

                self.D_optimizer.zero_grad()

                # D를 훈련할 때에는 실제 데이터와 fake 데이터의 loss를 각각 구한 후 한번에 backward & step
                D_real, C_real = self.D(x)
                D_real_loss = self.BCE_loss(D_real, y_real)
                C_real_loss = self.CE_loss(C_real, y)

                G_x = self.G(z, y_vec)
                D_fake, C_fake = self.D(G_x)
                D_fake_loss = self.BCE_loss(D_fake, y_fake)
                C_fake_loss = self.CE_loss(C_fake, y)

                discriminator_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                discriminator_loss.backward()
                self.D_optimizer.step()

                self.G_optimizer.zero_grad()

                if G_past is not None:
                    for k in range(self.class_index):
                        sample_z = torch.zeros(self.batch_size, self.noise_dim)
                        for i in range(self.batch_size // (self.class_index - 1)):
                            sample_z[i * (self.class_index - 1)] = torch.rand(1, self.noise_dim)
                            for j in range(1, self.class_index - 1):
                                sample_z[i * (self.class_index - 1) + j] = sample_z[i * (self.class_index - 1)]

                        temp = torch.zeros(self.class_index - 1, 1)
                        for i in range(self.class_index - 1):
                            temp[i, 0] = i

                        temp_y = torch.zeros(self.batch_size, 1)
                        for i in range(self.batch_size // (self.class_index - 1)):
                            temp_y[i * (self.class_index - 1): (i + 1) * (self.class_index - 1)] = temp

                        sample_y = torch.zeros(self.batch_size, self.total_class_num).scatter_(
                            1, temp_y.type(torch.LongTensor), 1)

                        if self.gpu_mode:
                            sample_y, sample_z = sample_y.cuda(), sample_z.cuda()

                        g_from_G = self.G(sample_z, sample_y)
                        g_from_G_past = G_past(sample_z, sample_y)

                        ra_loss = self.MSE_loss(g_from_G, g_from_G_past)
                        ra_loss.backward()

                        # 이전 G와의 loss를 먼저 optimize
                        # self.G_optimizer.step()
                        # self.G_optimizer.zero_grad()

                        # code 원래 위치

                G_x = self.G(z, y_vec)
                D_fake, C_fake = self.D(G_x)
                generator_loss = self.BCE_loss(D_fake, y_real) + self.CE_loss(C_fake, y)

                generator_loss.backward()
                self.G_optimizer.step()

                if ((idx + 1) % 10) == 0:
                    #if self.method == 'joint_retraining' or self.class_index == 1:
                    if G_past is None:
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (idx + 1), self.data_loader.dataset.__len__() // self.batch_size,
                               discriminator_loss.item(), generator_loss.item()))
                    #elif self.method == 'replay_alignment':
                    else:
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss_with_G_past: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (idx + 1), self.data_loader.dataset.__len__() // self.batch_size,
                               discriminator_loss.item(), ra_loss.item(), generator_loss.item()))

            with torch.no_grad():
                self.visualize_results((epoch + 1))

    def visualize_results(self, epoch):
        self.G.eval()

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        # class_index: class_index 대신 현재까지 훈련한 class의 수를 전달해야
        for i in range(self.class_index):
            if i == 0:
                y = torch.randint(i, i + 1, (10, 1))
            else:
                y = torch.cat([y, torch.randint(i, i + 1, (10, 1))])

        sample_y = torch.zeros(self.class_index * 10, 10).scatter_(
            1, y.type(torch.LongTensor), 1)
        sample_z_ = torch.rand((self.class_index * 10, self.noise_dim))

        if self.gpu_mode:
            sample_z_, sample_y = sample_z_.cuda(), sample_y.cuda()

        samples = self.G(sample_z_, sample_y)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        images = np.squeeze(
            self.merge(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim]))
        # result_dir = 'result/MNIST'
        # class_index: 몇 개의 class를 훈련시켰는지에 대한 변수 대신 사용
        imageio.imwrite(self.result_dir + '/' + 'to_%d' % (self.class_index - 1) + '/' + '_epoch%03d' % epoch + '.png', images)

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        if (images.shape[3] in (3, 4)):
            c = images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w, :] = image
            return img
        elif images.shape[3] == 1:
            img = np.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
            return img
        else:
            raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from ACGAN import Generator


class TestGenerator(object):
    def __init__(self, args):
        self.method = args.method
        self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
        self.classifier.load_state_dict(torch.load('./network/resnet18.pt'))
        self.generator = Generator()
        self.task = args.task
        self.gpu_mode = torch.cuda.is_available()
        self.batch_size = args.batch_size
        self.noise_dim = 100
        self.total_class_num = 10

        if args.dataset == 'MNIST':
            self.classifier.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.task == 'to_4':
            self.generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_4.pt'))
        elif self.task == 'to_9':
            self.generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_9.pt'))

    def test(self):
        correct = 0
        self.classifier.eval()
        self.generator.eval()

        if self.gpu_mode:
            self.classifier, self.generator = self.classifier.cuda(), self.generator.cuda()

        for i in range(5000):
            noise = torch.rand(self.batch_size, self.noise_dim)

            # one_hot vector
            if self.task == 'to_4':
                one_hot_label = torch.randint(0, 5, [self.batch_size])
            elif self.task == 'to_9':
                one_hot_label = torch.randint(0, 10, [self.batch_size])

            # answer labels
            labels = torch.zeros(self.batch_size, 10).scatter_(  # batch_size, total_class_num
                1, one_hot_label.type(torch.LongTensor).unsqueeze(1), 1)

            if self.gpu_mode:
                noise, labels, one_hot_label = noise.cuda(), labels.cuda(), one_hot_label.cuda()

            generated_images = self.generator(noise, labels)

            output = self.classifier(generated_images)
            prediction = output.data.max(1, keepdim=True)[1]
            if torch.cuda.is_available():
                correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().cpu().sum()
            else:
                correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().sum()

            if (i + 1) % 20 == 0:
                print("Accuracy: {}/{} ({:.3f}%)".format(correct, (i + 1) * 64, 100. * correct / ((i + 1) * 64)))

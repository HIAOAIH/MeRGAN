import torch
import torch.nn as nn
from torchvision import datasets, transforms
from ACGAN import Generator


class TestGenerator(object):
    def __init__(self, args):
        self.method = args.method
        self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
        self.generator = Generator()
        self.task = args.task
        self.gpu_mode = torch.cuda.is_available()
        self.batch_size = args.batch_size
        self.noise_dim = 100
        self.total_class_num = 10

        if args.dataset == 'MNIST':
            self.classifier.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.task == 'to_4':
            self.classifier.load_state_dict(torch.load('./network/resnet18_to_4.pt'))
            self.generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_4.pt'))
        elif self.task == 'to_9':
            self.classifier.load_state_dict(torch.load('./network/resnet18_to_9.pt'))
            self.generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_9.pt'))

    def test(self):
        correct = 0

        if self.gpu_mode:
            classifier, generator = self.classifier.cuda(), self.generator.cuda()

        for i in range(2000):
            noise = torch.rand(self.batch_size, self.noise_dim)
            # one_hot vector
            one_hot_label = torch.randint(0, 10, [self.batch_size])
            # answer labels
            labels = torch.zeros(self.batch_size, 10).scatter_(  # batch_size, total_class_num
                1, one_hot_label.type(torch.LongTensor).unsqueeze(1), 1)

            if self.gpu_mode:
                noise, labels, one_hot_label = noise.cuda(), labels.cuda(), one_hot_label.cuda()

            generated_images = generator(noise, labels)

            output = classifier(generated_images)
            prediction = output.data.max(1, keepdim=True)[1]
            if torch.cuda.is_available():
                correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().cpu().sum()
            else:
                correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().sum()

            if i % 20 == 0:
                print("Accuracy: {}/{} ({:.3f}%)".format(correct, (i + 1) * 64, 100. * correct / ((i + 1) * 64)))


#
# classifier = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
# classifier.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# classifier.load_state_dict(torch.load('./network/resnet18_to_9.pt'))
#
# generator = Generator()
# generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_9.pt'))
# # generator.load_state_dict(torch.load('./network/jrt/generator_jrt_to_4.pt'))
#
# correct = 0
#
# if torch.cuda.is_available():
#     classifier, generator = classifier.cuda(), generator.cuda()
#
# for i in range(2000):
#     noise = torch.rand(64, 100) # batch_size, noise_dim
#     # one_hot vector
#     one_hot_label = torch.randint(0, 10, [64])
#     # answer labels
#     labels = torch.zeros(64, 10).scatter_(  # batch_size, total_class_num
#         1, one_hot_label.type(torch.LongTensor).unsqueeze(1), 1)
#
#     if torch.cuda.is_available():
#         noise, labels, one_hot_label = noise.cuda(), labels.cuda(), one_hot_label.cuda()
#
#     generated_images = generator(noise, labels)
#
#     output = classifier(generated_images)
#     prediction = output.data.max(1, keepdim=True)[1]
#     if torch.cuda.is_available():
#         correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().cpu().sum()
#     else:
#         correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().sum()
#
#     if i % 20 == 0:
#         print("Accuracy: {}/{} ({:.3f}%)".format(correct, (i + 1) * 64, 100. * correct / ((i + 1) * 64)))


# for i in range(2000):
#     noise = torch.rand(64, 100) # batch_size, noise_dim
#     # one_hot vector
#     one_hot_label = torch.randint(0, 5, [64])
#     # one_hot_label = torch.randint(0, 10, [64])
#
#     # answer labels
#     labels = torch.zeros(64, 10).scatter_(  # batch_size, total_class_num
#         1, one_hot_label.type(torch.LongTensor).unsqueeze(1), 1)
#
#     if torch.cuda.is_available():
#         noise, labels, one_hot_label = noise.cuda(), labels.cuda(), one_hot_label.cuda()
#
#     generated_images = generator(noise, labels)
#
#     output = classifier(generated_images)
#     prediction = output.data.max(1, keepdim=True)[1]
#     if torch.cuda.is_available():
#         correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().cpu().sum()
#     else:
#         correct += prediction.eq(one_hot_label.data.view_as(prediction)).long().sum()
#
#     if i % 20 == 0:
#         print("Accuracy: {}/{} ({:.0f}%)".format(correct, (i + 1) * 64, 100. * correct / ((i + 1) * 64)))
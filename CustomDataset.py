import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        super(CustomDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.len = len(self.targets)

    def __getitem__(self, index):
        return self.data[index].view(1, 28, 28), self.targets[index]

    def __len__(self):
        return self.len

    def concat_datasets(self, concated_dataset):
        for idx, (x, y) in enumerate(concated_dataset):
            if idx == 0:
                tmp_data = x
                tmp_targets = torch.tensor([y])
            else:
                tmp_data = torch.cat((tmp_data, x), 0)
                tmp_targets = torch.cat((tmp_targets, torch.tensor([y])), 0)

        self.data = torch.cat((self.data, tmp_data), 0)
        self.targets = torch.cat((self.targets, tmp_targets), 0)
        self.len = self.len + len(concated_dataset.targets)

    def append(self, data, targets):
        self.data = torch.cat((self.data, data), 0)
        self.targets = torch.cat((self.targets, targets), 0)
        self.len = self.len + len(targets)
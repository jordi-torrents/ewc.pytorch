import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):
    def __init__(self, train: bool, permute_idx: torch.Tensor):
        super().__init__("~/.torch/data/mnist", train, download=True)
        assert len(permute_idx) == 28 * 28
        self.data = (self.data / 255).view(len(self.data), -1)[:, permute_idx]

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def get_sample(self, sample_size: int):
        return self.data[torch.randperm(len(self.data))[:sample_size]]

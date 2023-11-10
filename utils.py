from typing import Iterable

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.nn import functional as F


class EWC:
    def __init__(self, model: nn.Module, dataset: torch.Tensor):
        self.means = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.precision_matrices = self.diag_fisher(model, dataset)

    @staticmethod
    def diag_fisher(model: nn.Module, dataset: torch.Tensor):
        precision_matrices: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            precision_matrices[name] = torch.zeros_like(parameter)

        model.eval()

        output: torch.Tensor = model(dataset.cuda())
        labels = output.max(1).indices
        loss = F.nll_loss(F.log_softmax(output, dim=1), labels, reduction="none")

        for loss_per_image in loss:
            model.zero_grad()
            loss_per_image.backward(retain_graph=True)

            for name, parameter in model.named_parameters():
                assert parameter.requires_grad
                assert parameter.grad is not None
                precision_matrices[name] += parameter.grad**2

        for precision_matrix in precision_matrices.values():
            precision_matrix /= len(dataset)

        return precision_matrices

    def penalty(self, model: nn.Module):
        return sum(
            (self.precision_matrices[name] * (parameter - self.means[name]) ** 2).sum()
            for name, parameter in model.named_parameters()
        )


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ewc: EWC | None = None,
    importance: float = 1,
):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        optimizer.zero_grad()
        output = model(input.cuda())
        loss = F.cross_entropy(output, target.cuda())
        if ewc is not None:
            loss += importance * ewc.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)  # type: ignore


def test(model: nn.Module, dataset: torchvision.datasets.MNIST):
    model.eval()

    with torch.no_grad():
        predctions: torch.Tensor = model(dataset.data.cuda()).max(dim=1).indices
        return (
            (predctions == dataset.targets.cuda()).count_nonzero()
            / len(dataset.targets)
        ).item()

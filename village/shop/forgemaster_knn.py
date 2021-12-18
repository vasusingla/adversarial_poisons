"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss, reverse_xent_avg
import pdb
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterKNN(_Forgemaster):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)[0]
            # outputs_normalized = (outputs - self.feature_mean)/self.feature_std
            # outputs_normalized = outputs_normalized/(outputs_normalized.norm(dim=-1, keepdim=True)+1e-8)
            # print(type(labels), labels.shape)
            # print(type(outputs), outputs.shape)
            loss = -torch.cdist(outputs.unsqueeze(dim=1), labels).sum()
            loss.backward(retain_graph=self.retain)

            # bogus prediction vector
            prediction = torch.zeros(1)

            return loss.detach().cpu(), prediction.detach().cpu()

        return closure
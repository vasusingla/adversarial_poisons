"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss, reverse_xent_avg
import pdb
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterKNN_farthest(_Forgemaster):
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
            outputs_normalized = (outputs)/outputs.norm(dim=1)[:, None]
            dot_prod = torch.einsum('bf,bkf->bk',outputs_normalized, labels)
            # print(dot_prod, labels, outputs_normalized)
            loss = (1-dot_prod).mean()
            # loss = -torch.cdist(outputs.unsqueeze(dim=1), labels.float()).sum()
            loss.backward(retain_graph=self.retain)
            # bogus prediction vector
            prediction = torch.zeros(1)
            return loss.detach().cpu(), prediction.detach().cpu()

        return closure
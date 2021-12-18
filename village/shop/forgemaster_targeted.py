"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterTargeted(_Forgemaster):

    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)
            new_labels = self._label_map(outputs, labels)
            if type(outputs) == list:
                losses = [criterion(output, new_label) for output, new_label
                          in zip(outputs, new_labels)]
                loss = sum(losses)
            else:
                loss = criterion(outputs, new_labels)
            loss.backward(retain_graph=self.retain)
            if type(outputs) == list:
                prediction = [(output.data.argmax(dim=1)==new_label).sum() for
                              output, new_label in zip(outputs, new_labels)]
                prediction = sum(prediction)/len(prediction)
            else:
                prediction = (outputs.data.argmax(dim=1) == new_labels).sum()
            # prediction = (outputs.data.argmax(dim=1) == new_labels).sum()
            return loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _label_map(self, outputs, labels):
        # This is a naiive permutation on the label space. You can implement
        # any permutation you like here.
        if type(labels)==list:
            new_labels = [(label+1)%output.shape[1] for label, output in zip(labels, outputs)]
        else:
            new_labels = (labels + 1) % outputs.shape[1]
        return new_labels

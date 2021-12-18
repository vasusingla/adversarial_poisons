"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
import pdb
import random
torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterKMeans(_Forgemaster):

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        super().__init__(args, setup)
        self.kmeans_centroids = torch.load(args.centroid_path).to(setup['device'])
        self.kmeans_centroids = self.kmeans_centroids/(self.kmeans_centroids.norm(dim=-1, keepdim=True)+1e-8)
        self.feature_mean = torch.load(args.feature_mean_path).to(setup['device'])
        self.feature_std = torch.load(args.feature_std_path).to(setup['device'])



    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)[0]
            outputs_normalized = (outputs - self.feature_mean)/self.feature_std
            outputs_normalized = outputs_normalized/(outputs_normalized.norm(dim=-1, keepdim=True)+1e-8)
            new_labels = self._label_map(labels)

            # become close to new centroid, chosen by random permutation
            sim_matrix = torch.mm(outputs_normalized, self.kmeans_centroids.T)
            _, sim_indices = sim_matrix.topk(k=1, dim=-1)
            cosine_dist_target_centroid = (1-sim_matrix.gather(dim=-1, index=new_labels)).mean()
            cosine_dist_target_centroid.backward(retain_graph=self.retain)

            # prediction is measured as closeness to new centroid
            prediction = (sim_indices.squeeze() == new_labels.squeeze()).sum()
            return cosine_dist_target_centroid.detach().cpu(), prediction.detach().cpu()
        return closure

    def _label_map(self, labels):
        # This is a naiive permutation on the label space. You can implement
        # any permutation you like here.
        new_labels = (labels + 1) % self.kmeans_centroids.shape[0]
        new_labels = new_labels.unsqueeze(1)
        return new_labels

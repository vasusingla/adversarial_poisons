"""Interface for poison recipes."""
from .forgemaster_untargeted import ForgemasterUntargeted
from .forgemaster_targeted import ForgemasterTargeted
from .forgemaster_explosion import ForgemasterExplosion
from .forgemaster_tensorclog import ForgemasterTensorclog
from .forgemaster_knn import ForgemasterKNN
from .forgemaster_kmeans import ForgemasterKMeans
from .forgemaster_knn_farthest import ForgemasterKNN_farthest

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'grad_explosion':
        return ForgemasterExplosion(args, setup)
    elif args.recipe == 'tensorclog':
        return ForgemasterTensorclog(args, setup)
    elif args.recipe == 'untargeted':
        return ForgemasterUntargeted(args, setup)
    elif args.recipe == 'targeted':
        return ForgemasterTargeted(args, setup)
    elif args.recipe == 'knn':
        return ForgemasterKNN(args, setup)
    elif args.recipe == 'knn_farthest':
        return ForgemasterKNN_farthest(args, setup)
    elif args.recipe == 'kmeans':
        return ForgemasterKMeans(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']

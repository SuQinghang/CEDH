import torch
import torch.nn.functional as F



def cosine_S(training_targets, train_targets):
    '''
    Use cosine similarity to generate S
    '''
    training_targets_n = F.normalize(training_targets)
    train_targets_n = F.normalize(train_targets)
    S = torch.mm(training_targets_n, train_targets_n.t())
    return S

def smooth_S(training_targets, train_targets, eta):
    '''
    Smoothed S
    '''
    S = (training_targets @ train_targets.t() > 0).float()
    S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
    r = S.sum() / (1 - S).sum()
    
    S = S * (eta + r) - r 
    return S

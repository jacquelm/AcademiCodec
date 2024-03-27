import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import fairseq
import joblib

class DistillationLoss_cont(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, feature, teacher_cont, mask):
        '''
        feature: (bs, timesteps, 768)
        teacher_cont: (bs, timesteps, 768)
        '''
        
        T = feature.size(1)
        feature_norm = feature / feature.norm(dim=-1, keepdim=True)
        teacher_cont_norm = teacher_cont / teacher_cont.norm(dim=-1, keepdim=True)
        cos_similarity = einsum('b t d, b t d -> b t', feature_norm, teacher_cont_norm)
        
        if mask is not None:
            cos_similarity = cos_similarity.masked_fill(~mask, 0)
        s = torch.sigmoid(cos_similarity)
        loss = torch.log(torch.sigmoid(cos_similarity)).sum(dim=1) / T
        loss = loss.mean()
        return loss 
    
class DistillationLoss_pseudo(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
      
    def forward(self, feature, teacher_unit, mask):
        '''
        feature: (bs, timesteps, 768),
        teacher_unit: (bs, timesteps)
        '''
        feature = feature.permute(0, 2, 1)
        teacher_unit = teacher_unit.long()
        if mask is not None:
            teacher_unit = teacher_unit.masked_fill(~mask, -1)
        loss = F.cross_entropy(feature, teacher_unit, ignore_index=-1)
        return loss 
    
def loss_distillation(feature, teacher_cont, teacher_unit, device):
    
    '''
    teacher_cont: (bs, timesteps, 768)
    teacher_unit: (bs, 74)
    feature: (bs, timesteps, 768)
    '''

    bs, feature_len, _ = feature.size()
    _, teacher_cont_len, _ = teacher_cont.size()
    _, teacher_unit_len = teacher_unit.size()
    cont_mask = None
    unit_mask = None
    
    if feature_len != teacher_cont_len:
        max_cont_len = max(feature_len, teacher_cont_len)
        cont_mask = torch.ones(bs, max_cont_len, dtype=torch.bool).to(device)
        if feature_len < max_cont_len:
            padding_size = max_cont_len - feature_len
            feature = F.pad(feature, (0, 0, 0, padding_size))
            cont_mask[:, feature_len:] = 0
        else:
            padding_size = max_cont_len - teacher_cont_len
            teacher_cont = F.pad(teacher_cont, (0, 0, 0, padding_size))
            cont_mask[:, teacher_cont_len:] = 0
            
    if feature_len != teacher_unit_len:
        max_unit_len = max(feature_len, teacher_unit_len)
        unit_mask =  torch.ones(bs, max_unit_len, dtype=torch.bool).to(device)
        if feature_len < max_unit_len:
            padding_size = max_unit_len - feature_len
            feature = F.pad(feature, (0, 0, 0, padding_size))
            unit_mask[:, feature_len:] = 0
        else:
            padding_size = max_unit_len - teacher_unit_len
            teacher_unit = F.pad(teacher_unit, (0, padding_size))
            unit_mask[:, teacher_unit_len:] = 0

    distillation_cont = DistillationLoss_cont(device)
    distillation_pseudo = DistillationLoss_pseudo(device)
    
    cont_loss = distillation_cont(feature, teacher_cont, cont_mask)
    pseudo_loss = distillation_pseudo(feature, teacher_unit, unit_mask)

    return cont_loss ,pseudo_loss
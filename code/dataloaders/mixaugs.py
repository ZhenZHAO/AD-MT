import numpy as np
import random
import torch
import scipy.stats as stats


# # # # # # # # # # # # # # # # # # # # # 
# # 0. random box
# # # # # # # # # # # # # # # # # # # # # 
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1. cutmix for 2d
# # # # # # # # # # # # # # # # # # # # # 
# def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()
    
#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
#     # print(u_rand_index)
    
#     # get box
#     u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         # label is of 3 dimensions
# #         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
# #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # 
# # 2. cutmix for 3d
# # # # # # # # # # # # # # # # # # # # # 

def rand_bbox_3d(size, lam=None):
    # img: B x C x H x W x D, lb: B x H x W x D
    if len(size) == 5:
        W = size[2]
        H = size[3]
        D = size[4]
    elif len(size) == 4:
        W = size[1]
        H = size[2]
        D = size[3]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    cz = np.random.randint(size=[B, ], low=int(D/8), high=D)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)


    return bbx1, bby1, bbz1, bbx2, bby2, bbz2



# def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()
    
#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

#     # get box
#     # img: B x C x H x W x D, lb: B x H x W x D
#     u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits




def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits

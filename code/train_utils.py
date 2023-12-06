import os
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from utils.util import update_values, time_str, AverageMeter


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  1. alternate updating
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class AlternateUpdate:
    def __init__(self, alternate_period, initial_flag=True, flag_random=False):
        self.alternate_period = alternate_period
        self.flag_alternate = initial_flag
        self._counter = 0
        self.flag_random = flag_random
        if self.flag_random:
            self.random_alternate_period = np.random.randint(1, self.alternate_period +1)
        else:
            self.random_alternate_period = self.flag_alternate

    def reset(self, alternate_period, initial_flag=True, flag_random=False):
        self._counter = 0
        self.alternate_period = alternate_period
        self.flag_alternate = initial_flag
        self.flag_random = flag_random

        if self.flag_random:
            self.random_alternate_period = np.random.randint(1, self.alternate_period +1)
        else:
            self.random_alternate_period = self.flag_alternate

    def get_alternate_state(self):
        return self.flag_alternate

    def get_alternate_period(self):
        return self.alternate_period
    
    def set_alternate_period(self, new_period):
        if new_period > 0:
            self.alternate_period = new_period
            if self.flag_random:
                self.random_alternate_period = np.random.randint(1, self.alternate_period +1)
            else:
                self.random_alternate_period = self.flag_alternate

    # def set_alternate_period(self, new_period):
    #     if new_period > 0 and new_period != self.get_alternate_period():
    #         self.alternate_period = new_period
    #         self.random_alternate_period = np.random.randint(1, self.alternate_period +1)
    
    def update(self):
        # assert self._counter < self.alternate_period, f"{self._counter}/{self.alternate_period}"
        self._counter += 1
        if self._counter >= self.random_alternate_period:
            self.flag_alternate = not self.flag_alternate
            self._counter = 0
            if self.flag_random:
                self.random_alternate_period = np.random.randint(1, self.alternate_period +1)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  2. calculate unsupervised loss from two teachers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_unsupervised_loss_by_2teachers(
    predict, 
    target1, logits1, 
    target2, logits2, 
    entropy1=None, entropy2=None,
    weight_conflict=1.0, 
    mode_conflict="latest",
    flag_t1_update_latest=True, 
    thresh=0.95):

    batch_size, num_class, h, w = predict.shape
    
    # ----------
    # remove ent_fusion, kinda complicate, but similar performance to pixel confidence
    # ----------
    # dealing with conflicts and obtain results
    target, logits, mtx_bool_conflict = get_compromise_pseudo_after_conflict(target1, logits1, target2, logits2, mode_conflict, flag_t1_update_latest, num_class, entropy1, entropy2)
    
    # final calculations
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    target_consist = target.clone()
    target_consist[mtx_bool_conflict] = 255
    target_conflct = target.clone()
    target_conflct[~mtx_bool_conflict] = 255

    loss_consist = F.cross_entropy(predict, target_consist, ignore_index=255, reduction="none")
    loss_conflct = F.cross_entropy(predict, target_conflct, ignore_index=255, reduction="none")
    loss = loss_consist + loss_conflct * weight_conflict

    # return conflicts
    # return loss.sum() / thresh_mask.float().sum(), mtx_bool_conflict.float().sum() // batch_size
    return loss.mean(), mtx_bool_conflict.float().sum() // batch_size


def get_compromise_pseudo_after_conflict(target1, logits1, target2, logits2, 
        mode_conflict, flag_t1_update_latest, num_cls, entropy1=None, entropy2=None):
    target = target1.clone()
    logits = logits1.clone()
    mtx_bool_conflict = target1 != target2

    if "low_ent" in mode_conflict:
        if entropy1 is not None and entropy2 is not None:
            if "low_ent_all" in mode_conflict:
                tmp_flag = entropy2.sum(dim=[1,2]) < entropy1.sum(dim=[1,2])
            else: # "low_ent_conflict" --> only the conflict region
                tmp_flag = (entropy2 * mtx_bool_conflict.float()).sum(dim=[1,2]) < (entropy1 * mtx_bool_conflict.float()).sum(dim=[1,2]) 
            target[tmp_flag, :, :] = target2[tmp_flag, :, :]
            logits[tmp_flag, :, :] = logits[tmp_flag, :, :]
    
    elif "latest" in mode_conflict:
        if not flag_t1_update_latest:
            target[mtx_bool_conflict] = target2[mtx_bool_conflict]
            logits[mtx_bool_conflict] = logits2[mtx_bool_conflict]
    
    elif "random" in mode_conflict:
        if np.random.random() < 0.5:
            target[mtx_bool_conflict] = target2[mtx_bool_conflict]
            logits[mtx_bool_conflict] = logits2[mtx_bool_conflict]

    elif "pixel_confidence" in mode_conflict:
        if entropy1 is not None and entropy2 is not None:
            bool_better_tea2 = entropy2 < entropy1
        else:
            bool_better_tea2 = logits2 > logits1
        target[bool_better_tea2] = target2[bool_better_tea2]
        logits[bool_better_tea2] = logits2[bool_better_tea2]

    else:
        raise NotImplementedError(
            "conflict mode {} is not supported".format(mode_conflict)
        )

    return target, logits, mtx_bool_conflict



def get_compromise_pseudo_btw_tea_stu(target_tea, logits_tea, target_stu, logits_stu, 
                                      mode_conflict, 
                                      mtx_teacher_conflict=None):
    target = target_tea.clone()
    logits = logits_tea.clone()
    if mtx_teacher_conflict is None:
        mtx_bool_conflict = target_tea != target_stu
    else:
        mtx_bool_conflict_stu = target_tea != target_stu
        mtx_bool_conflict = mtx_bool_conflict_stu & mtx_teacher_conflict

    
    if "random" in mode_conflict:
        if np.random.random() < 0.5:
            target[mtx_bool_conflict] = target_stu[mtx_bool_conflict]
            logits[mtx_bool_conflict] = logits_stu[mtx_bool_conflict]

    elif "pixel_confidence" in mode_conflict:
        bool_better_stu = logits_stu > logits_tea
        bool_better_stu_select = bool_better_stu & mtx_bool_conflict
        target[bool_better_stu_select] = target_stu[bool_better_stu_select]
        logits[bool_better_stu_select] = logits_stu[bool_better_stu_select]

    
    elif "always_tea" in mode_conflict:
        pass

    elif "always_stu" in mode_conflict:
        target[mtx_bool_conflict] = target_stu[mtx_bool_conflict]
        logits[mtx_bool_conflict] = logits_stu[mtx_bool_conflict]

    else:
        raise NotImplementedError(
            "conflict mode {} is not supported".format(mode_conflict)
        )

    return target, logits, mtx_bool_conflict



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  3. calculate differencens between teachers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def check_diffs_between_models(
    model1,
    model2,
    data_loader,
    epoch,
    logger,
    cfg
):
    model1.eval()
    model2.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )

    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()

    for _, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            output1, _ = model1(images)
            output2, _ = model2(images)

        # get the output produced by model_teacher
        output1 = output1.data.max(1)[1].cpu().numpy()
        output2 = output2.data.max(1)[1].cpu().numpy()

        # start to calculate miou
        intersection1, union1, _ = intersectionAndUnion(
            output1, output2, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection1 = torch.from_numpy(intersection1).cuda()
        reduced_union1 = torch.from_numpy(union1).cuda()

        intersection_meter1.update(reduced_intersection1.cpu().numpy())
        union_meter1.update(reduced_union1.cpu().numpy())

    iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
    mIoU1 = np.mean(iou_class1)

    return mIoU1


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

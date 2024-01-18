import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml


import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from dataloaders.mixaugs import cut_mix
from dataloaders.dataset_2d import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from train_utils import AlternateUpdate, get_compromise_pseudo_btw_tea_stu
from val_2D import test_single_volume


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0 
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)
    
    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    # update buffers
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    base_lr = args["base_lr"]
    num_classes = args["num_classes"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    cur_time = time_str()
    writer = SummaryWriter(snapshot_path + '/log')
    csv_train = os.path.join(snapshot_path, "log", "seg_{}_train_iter.csv".format(cur_time))
    csv_test = os.path.join(snapshot_path, "log", "seg_{}_validate_ep.csv".format(cur_time))

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args["model"], in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model = create_model()
    ema_model = create_model(ema=True)
    ema_model_another = create_model(ema=True)
    model.cuda()
    ema_model.cuda()
    ema_model_another.cuda()
    model.train()
    ema_model.train()
    ema_model_another.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    db_train = BaseDataSets(base_dir=args["root_path"], split="train", num=None, 
                                transform=transforms.Compose([WeakStrongAugment(args["patch_size"])])
                                )
    db_val = BaseDataSets(base_dir=args["root_path"], split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args["root_path"], args["labeled_num"])
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    if args.get("flag_sampling_based_on_lb", False):
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, batch_size-args["labeled_bs"])
    else:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"])

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance_another = 0.0
    best_performance_stu = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # alternate params
    alt_flag_epoch_shuffle_teachers = args["alt_flag_epoch_shuffle_teachers"]
    alt_flag_conflict_mode = args["alt_flag_conflict_mode"]
    alt_flag_conflict_stu_use_more = args["alt_flag_conflict_stu_use_more"]
    alt_param_ensemble_temp = args["alt_param_ensemble_temp"]

    alt_param_conflict_weight = args["alt_param_conflict_weight"]
    alt_param_updating_period_iters = args["alt_param_updating_period_iters"]
    alt_flag_updating_period_random = args["alt_flag_updating_period_random"]
    
    alt_param_updating_period_iters = args["alt_param_updating_period_iters"]

    alternate_indicator = AlternateUpdate(alt_param_updating_period_iters, 
                                          initial_flag=True, flag_random=alt_flag_updating_period_random)
    for epoch_num in iterator:
        # update alt_params
        # a) alternate flag
        if alt_flag_epoch_shuffle_teachers:
            alternate_indicator.reset(alt_param_updating_period_iters, 
                                      initial_flag=(epoch_num % 2 == 0), 
                                      flag_random=alt_flag_updating_period_random)
        # b) conflict weight
        var_param_conflict_weight = alt_param_conflict_weight
        # c) flag of starting self training
        flag_start_self_learning = False

        # metric indicators
        meter_sup_losses = AverageMeter()
        meter_uns_losses = AverageMeter(20)
        meter_train_losses = AverageMeter(20)
        meter_learning_rates = AverageMeter()
        meter_highc_ratio = AverageMeter()
        meter_conflict_ratio = AverageMeter()
        meter_uns_losses_consist = AverageMeter(20)
        meter_uns_losses_conflict = AverageMeter(20)

        for i_batch, sampled_batch in enumerate(trainloader):
            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            # get batched data
            if args.get("flag_sampling_based_on_lb", False):
                img_lb_w, target_lb = weak_batch[:num_lb], label_batch[:num_lb]
                img_ulb_w, img_ulb_s = weak_batch[num_lb:], strong_batch[num_lb:]
            else:
                img_lb_w, target_lb = weak_batch[num_ulb:], label_batch[num_ulb:]
                img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]

            # 2) getting pseudo labels
            loss_ulb_consist, loss_ulb_conflict = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            with torch.no_grad():
                if not flag_start_self_learning and not args["flag_pseudo_from_student"]:
                    if alternate_indicator.get_alternate_state():
                        ema_model.train()
                        ema_model_another.eval()
                    else:
                        ema_model_another.train()
                        ema_model.eval()

                    
                    ema_outputs_soft_1 = torch.softmax(ema_model(img_ulb_w), dim=1)
                    ema_outputs_soft_1 = ema_outputs_soft_1.detach()

                    ema_outputs_soft_2 = torch.softmax(ema_model_another(img_ulb_w), dim=1)
                    ema_outputs_soft_2 = ema_outputs_soft_2.detach()

                    _, pseudo_outputs_1 = torch.max(ema_outputs_soft_1, dim=1)
                    _, pseudo_outputs_2 = torch.max(ema_outputs_soft_2, dim=1)

                    mtx_bool_conflict = pseudo_outputs_1 != pseudo_outputs_2
                    conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb

                    # entropy
                    entropy_1 = -torch.sum(ema_outputs_soft_1 * torch.log2(ema_outputs_soft_1 + 1e-10), dim=1)
                    entropy_2 = -torch.sum(ema_outputs_soft_2 * torch.log2(ema_outputs_soft_2 + 1e-10), dim=1)

                    # weighted sum
                    weights_1 = torch.exp(-entropy_1) / (torch.exp(-entropy_1) + torch.exp(-entropy_2))
                    weights_2 = 1.0 - weights_1
                    weighted_outputs = weights_1.unsqueeze(1) * ema_outputs_soft_1 + weights_2.unsqueeze(1) * ema_outputs_soft_2


                    weighted_outputs = torch.pow(weighted_outputs, 1.0 / alt_param_ensemble_temp)
                    weighted_outputs = weighted_outputs / torch.sum(weighted_outputs, dim=1, keepdim=True)
                    
                    # get final outputs
                    pseudo_logits, pseudo_outputs = torch.max(weighted_outputs, dim=1)
                    del ema_outputs_soft_1, pseudo_outputs_1, ema_outputs_soft_2, pseudo_outputs_2, weighted_outputs, entropy_1, entropy_2

                else:
                    model.eval()
                    ema_outputs_soft = torch.softmax(model(img_ulb_w), dim=1)
                    pseudo_logits, pseudo_outputs = torch.max(ema_outputs_soft.detach(), dim=1)
                    mtx_bool_conflict = None
                    conflict_ratio = torch.tensor(0.0).cuda()
                    model.train()

            # 3) apply cutmix
            if flag_start_self_learning: # strongest augs
                img_ulb_s, pseudo_outputs, pseudo_logits = cut_mix(
                                img_ulb_s,
                                pseudo_outputs,
                                pseudo_logits
                )
            else:
                if alternate_indicator.get_alternate_state() == False:
                    del img_ulb_s
                    if np.random.random() < args.get("cutmix_prob", 1.0):
                        if mtx_bool_conflict is None:
                            img_ulb_s, pseudo_outputs, pseudo_logits = cut_mix(
                                            img_ulb_w,
                                            pseudo_outputs,
                                            pseudo_logits
                            )
                        else:
                            img_ulb_s, pseudo_outputs, pseudo_logits, mtx_bool_conflict = cut_mix(
                                            img_ulb_w,
                                            pseudo_outputs,
                                            pseudo_logits,
                                            mtx_bool_conflict
                            )

                        conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb
                    else:
                        img_ulb_s = img_ulb_w.clone()

            # 4) forward
            img = torch.cat((img_lb_w, img_ulb_w, img_ulb_s))
            pred = model(img)
            pred_lb = pred[:args["labeled_bs"]]
            pred_ulb_w, pred_ulb_s = pred[args["labeled_bs"]:].chunk(2)

            with torch.no_grad():
                pseudo_logits_stu, pseudo_outputs_stu = torch.max(torch.softmax(pred_ulb_w.detach(), dim=1), dim=1)
            
            # 5) supervised loss
            loss_lb = (ce_loss(pred_lb, target_lb.long()) +
                        dice_loss(torch.softmax(pred_lb, dim=1),
                                target_lb.unsqueeze(1).float(), 
                                ignore=torch.zeros_like(target_lb).float())
                        ) / 2.0
            
            # 6) unsupervised loss
            pseudo_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
            high_ratio = pseudo_mask.float().mean()
            if "dice" == args["flag_ulb_loss_type"]:
                if flag_start_self_learning:
                    loss_ulb = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                    pseudo_outputs.unsqueeze(1).float(),
                                    ignore=(pseudo_logits < args["alt_param_threshold_self_training"]).float())
                else:
                    pseudo_outputs, pseudo_logits, conflict_tea_stu = get_compromise_pseudo_btw_tea_stu(pseudo_outputs, pseudo_logits, 
                                                                                        pseudo_outputs_stu, pseudo_logits_stu, 
                                                                                        alt_flag_conflict_mode, 
                                                                                        None if alt_flag_conflict_stu_use_more else mtx_bool_conflict)

                    if var_param_conflict_weight > 1 or var_param_conflict_weight < 1:
                        pseudo_logits_consist = pseudo_logits.clone()
                        pseudo_logits_consist[mtx_bool_conflict] = -0.1
                        loss_ulb_consist = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                                        pseudo_outputs.unsqueeze(1).float(),
                                                        ignore=(pseudo_logits_consist < args["conf_threshold"]).float())
                        if var_param_conflict_weight > 0 or var_param_conflict_weight < 0:
                            # print("#"*20, var_param_conflict_weight, pseudo_outputs.requires_grad, pseudo_logits_consist.requires_grad)
                            pseudo_logits_conflict = pseudo_logits.clone()
                            pseudo_logits_conflict[~mtx_bool_conflict] = -0.1
                            loss_ulb_conflict = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                                            pseudo_outputs.unsqueeze(1).float(),
                                                            ignore=(pseudo_logits_conflict < args["conf_threshold"]).float())
                            if loss_ulb_conflict > 0.0:
                                loss_ulb = loss_ulb_consist + var_param_conflict_weight * loss_ulb_conflict
                            else:
                                # print("-"*100, loss_ulb_conflict.item(), loss_ulb_consist.item())
                                loss_ulb = loss_ulb_consist * 1.0
                        else:
                            loss_ulb = loss_ulb_consist * 1.0

                    else:
                        loss_ulb = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                        pseudo_outputs.unsqueeze(1).float(),
                                        ignore=(pseudo_logits < args["conf_threshold"]).float())
            else:
                pseudo_outputs[~pseudo_mask] = -100
                loss_ulb = F.cross_entropy(pred_ulb_s, pseudo_outputs.long(), ignore_index=-100, reduction="mean")

            # 7) total loss
            consistency_weight = get_current_consistency_weight(iter_num//150, args)
            loss = loss_lb + consistency_weight * loss_ulb

            # 8) update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 9) update teacher model
            if not flag_start_self_learning:
                if alternate_indicator.get_alternate_state():
                    update_ema_variables(model, ema_model, args["ema_decay"], iter_num//2, args)
                else:
                    update_ema_variables(model, ema_model_another, args["ema_decay"], iter_num//2, args)
                
                alternate_indicator.update()
            else:
                update_ema_variables(model, ema_model, args["ema_decay"], iter_num, args)

            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = base_lr

            # 11) record statistics
            iter_num = iter_num + 1
            # --- a) writer
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_lb', loss_lb, iter_num)
            writer.add_scalar('info/loss_ulb', loss_ulb, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/conflict_ratio', conflict_ratio.item(), iter_num)
            # --- b) loggers
            logging.info("iteration:{}  t-loss:{:.4f}, loss-lb:{:.4f}, loss-ulb:{:.4f}, conflict/consist:{:.4f}/{:.4f}, weight:{:.2f}, high-r:{:.2f}, conflic:{}, lr:{:.4f}, alt:{}".format(iter_num, 
                loss.item(), loss_lb.item(), loss_ulb.item(), loss_ulb_conflict.item(), loss_ulb_consist.item(),
                consistency_weight, high_ratio, int(conflict_ratio.item()), lr_, alternate_indicator.get_alternate_state()))
            # --- c) avg meters
            meter_sup_losses.update(loss_lb.item())
            meter_uns_losses.update(loss_ulb.item())
            meter_train_losses.update(loss.item())
            meter_highc_ratio.update(high_ratio.item())
            meter_learning_rates.update(lr_)
            meter_conflict_ratio.update(conflict_ratio.item())
            meter_uns_losses_consist.update(loss_ulb_consist.item())
            meter_uns_losses_conflict.update(loss_ulb_conflict.item())

            # --- d) csv
            tmp_results = {
                        'loss_total': loss.item(),
                        'loss_lb': loss_lb.item(),
                        'loss_ulb': loss_ulb.item(),
                        'loss_ulb_conflict': loss_ulb_conflict.item(),
                        'loss_ulb_consist': loss_ulb_consist.item(),
                        'lweight_ub': consistency_weight,
                        'high_ratio': high_ratio.item(),
                        'conflict_ratio': conflict_ratio.item(),
                        "lr":lr_}
            data_frame = pd.DataFrame(data=tmp_results, index=range(iter_num, iter_num+1))
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode='a', header=None, index_label='iter')
            else:
                data_frame.to_csv(csv_train, index_label='iter')

            if iter_num >= max_iterations:
                break

        # 12) validating
        if epoch_num % args.get("test_interval_ep", 1) == 0 or iter_num >= max_iterations:
            model.eval()
            ema_model.eval()
            ema_model_another.eval()

            metric_list = 0.0
            ema_metric_list = 0.0
            ema_metric_list = 0.0
            ema_metric_another_list = 0.0

            for _, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"], 
                    sampled_batch["label"], 
                    model, 
                    classes=num_classes)
                metric_list += np.array(metric_i)

                ema_metric_i = test_single_volume(
                    sampled_batch["image"], 
                    sampled_batch["label"], 
                    ema_model, 
                    classes=num_classes)
                ema_metric_list += np.array(ema_metric_i)

                if not flag_start_self_learning:
                    ema_another_metric_i = test_single_volume(
                        sampled_batch["image"], 
                        sampled_batch["label"], 
                        ema_model_another, 
                        classes=num_classes)
                    ema_metric_another_list += np.array(ema_another_metric_i)

            metric_list = metric_list / len(db_val)
            ema_metric_list = ema_metric_list / len(db_val)
            if not flag_start_self_learning:
                ema_metric_another_list = ema_metric_another_list / len(db_val)

            for class_i in range(num_classes-1):
                writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], epoch_num)
                writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], epoch_num)

                writer.add_scalar('info/ema_val_{}_dice'.format(class_i+1), ema_metric_list[class_i, 0], epoch_num)
                writer.add_scalar('info/ema_val_{}_hd95'.format(class_i+1), ema_metric_list[class_i, 1], epoch_num)

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, epoch_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch_num)

            ema_performance = np.mean(ema_metric_list, axis=0)[0]
            ema_mean_hd95 = np.mean(ema_metric_list, axis=0)[1]
            writer.add_scalar('info/ema_val_mean_dice', ema_performance, epoch_num)
            writer.add_scalar('info/ema_val_mean_hd95', ema_mean_hd95, epoch_num)

            if not flag_start_self_learning:
                ema_another_performance = np.mean(ema_metric_another_list, axis=0)[0]
                ema_another_mean_hd95 = np.mean(ema_metric_another_list, axis=0)[1]
            else:
                ema_another_performance = 0.0
                ema_another_mean_hd95 = 0.0
            writer.add_scalar('info/ema_another_val_mean_dice', ema_another_performance, epoch_num)
            writer.add_scalar('info/ema_another_val_mean_hd95', ema_another_mean_hd95, epoch_num)

            if performance > best_performance_stu:
                best_performance_stu = performance
                tmp_stu_snapshot_path = os.path.join(snapshot_path, "student")
                if not os.path.exists(tmp_stu_snapshot_path):
                    os.makedirs(tmp_stu_snapshot_path,exist_ok=True)
                save_mode_path_stu = os.path.join(tmp_stu_snapshot_path, 'ep_{:0>3}_dice_{}.pth'.format(epoch_num, round(best_performance_stu, 4)))
                torch.save(model.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(snapshot_path,'{}_best_stu_model.pth'.format(args["model"]))
                torch.save(model.state_dict(), save_best_path_stu)


            if ema_performance > best_performance:
                best_performance = ema_performance
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path,exist_ok=True)
                save_mode_path = os.path.join(tmp_tea_snapshot_path, 'ep_{:0>3}_dice_{}.pth'.format(epoch_num, round(best_performance, 4)))
                torch.save(ema_model.state_dict(), save_mode_path)

                save_best_path = os.path.join(snapshot_path,'{}_best_tea_model.pth'.format(args["model"]))
                torch.save(ema_model.state_dict(), save_best_path)
            
            if ema_another_performance > best_performance_another:
                best_performance_another = ema_another_performance

            # csv
            tmp_results_ts = {
                    'loss_total': meter_train_losses.avg,
                    'loss_sup': meter_sup_losses.avg,
                    'loss_unsup': meter_uns_losses.avg,
                    'loss_unsup_consist': meter_uns_losses_consist.avg,
                    'loss_unsup_conflict': meter_uns_losses_conflict.avg,
                    'avg_high_ratio': meter_highc_ratio.avg,
                    'avg_conflict_ratio': meter_conflict_ratio.avg,
                    'learning_rate': meter_learning_rates.avg,
                    'Dice_tea': ema_performance,
                    'Dice_tea_best': best_performance,
                    'Dice_tea_another': ema_another_performance,
                    'Dice_tea_another_best': best_performance_another,
                    'Dice_stu': performance,
                    'Dice_stu_best': best_performance_stu}
            data_frame = pd.DataFrame(data=tmp_results_ts, index=range(epoch_num, epoch_num+1))
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_test, index_label='epoch')

            # logs
            logging.info(" <<Test>> - Ep:{}  - mean_dice/mean_h95 - S:{:.2f}/{:.2f}, Best-S:{:.2f}, T:{:.2f}/{:.2f}, Best-T:{:.2f}, T-a:{:.2f}/{:.2f}, Best-T-a:{:.2f}".format(epoch_num, 
                    performance*100, mean_hd95, best_performance_stu*100, ema_performance*100, ema_mean_hd95, best_performance*100, ema_another_performance*100, ema_another_mean_hd95, best_performance_another*100))
            logging.info("          - AvgLoss(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}, AvgConflict/consist:{:.4f}/{:.4f} highR:{:.2f}, conflict:{:.2f} ".format( 
                    meter_sup_losses.avg, meter_uns_losses.avg, meter_train_losses.avg, 
                    meter_uns_losses_conflict.avg, meter_uns_losses_consist.avg,
                    meter_highc_ratio.avg, meter_conflict_ratio.avg,
                    ))
            
            model.train()
            ema_model.train()
            ema_model_another.train()

        if (epoch_num+1) % args.get("save_interval_epoch", 1000000) == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='',
                        help='configuration file')
    
    # Basics: Data, results, model
    parser.add_argument('--root_path', type=str,
                        default='./data/ACDC', help='Name of Experiment')
    parser.add_argument('--res_path', type=str, 
                        default='./results/ACDC', help='Path to save resutls')
    parser.add_argument('--exp', type=str,
                        default='ACDC/POST-NoT', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of gpu used to train the model')
    
    # Training Basics
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument("--deterministic", action='store_true', 
                        help="whether use deterministic training")
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    parser.add_argument('--test_interval_ep', type=int,
                        default=1, help='')
    parser.add_argument('--save_interval_epoch', type=int,
                        default=1000000, help='')
    parser.add_argument("-p", "--poly", default=False, 
                        action='store_true', help="whether poly scheduler")
    

    # label and unlabel
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=136,
                        help='labeled data')
    
    # model related
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument("--flag_pseudo_from_student", default=False, 
                        action='store_true', help="using pseudo from student itself")
    
    # augmentation
    parser.add_argument('--cutmix_prob', type=float,  
                        default=0.5, help='probability of applying cutmix')
    
    # unlabeled loss
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=150.0, help='consistency_rampup')
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.95,
        help="confidence threshold for using pseudo-labels",
    )
    parser.add_argument('--flag_ulb_loss_type', type=str,
                        default="dice", help='loss type, ce, dice, dice+ce')
    parser.add_argument("--flag_sampling_based_on_lb", 
                        default=False, action='store_true', help="using dynamic cutmix")
    
    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args['cfg']
    cfgs_file = os.path.join('./cfgs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    # print confg information
    import pprint
    # print("{}".format(pprint.pformat(args)))
    # assert 1==0, "break here"

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    if args["gpu_id"] in range(10):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"], args["exp"], args["labeled_num"], args["model"])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)

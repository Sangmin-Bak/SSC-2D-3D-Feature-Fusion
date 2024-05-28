import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics.classification import Dice
from mmdet.models.losses.focal_loss import FocalLoss
from torch.cuda.amp import autocast

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 255
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="mean")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    with autocast(False):
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )

def precision_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
    )

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    with autocast(False):
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = pred[:, i, :, :, :]

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    

                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count

def CE_ssc_loss(pred, target, class_weights):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean

def Focal_ssc_loss(pred, target, class_weight, use_sigmoid=True, gamma=2.0, alpha=0.25):
    criterion = FocalLoss(use_sigmoid,
                          gamma,
                          alpha,
                          reduction="mean",
                          loss_weight=class_weight)
    num_classes = pred.shape[1]
    pred_valid = pred.permute(0, 2, 3, 4, 1)[target!=255, :]
    target_valid = target[target!=255]
    loss_valid = criterion(pred_valid.reshape(-1, num_classes), target_valid.reshape(-1).long())
    # loss_valid = loss[target!=255]
    # loss_valid_mean = torch.mean(loss_valid)
    return loss_valid


# def Dice_loss(pred, target):
#     # pred = F.softmax(pred, dim=1)
#     # occ_pred = torch.argmax(pred, dim=1)
#     # ones = torch.ones_like(occ_pred).to(occ_pred.device)
#     # occ_pred = torch.where(torch.logical_or(occ_pred==255, occ_pred==0), occ_pred, ones)

#     # ones = torch.ones_like(target).to(target.device)
#     # occ_target = torch.where(torch.logical_or(target==255, target==0), target, ones)

#     # occ_pred = occ_pred[target!=255]
#     # occ_target = occ_target[target!=255]

#     criterion = Dice(average='micro')
#     dice_score = criterion(pred, target.long())
#     loss = 1 - dice_score
#     return loss


def disillation_loss(pred, soft_label, target, T=10):
    # bs, h, w, z, c = pred.shape
    # pred = pred.permute(0, 2, 3, 4, 1)
    # soft_label = soft_label.permute(0, 2, 3, 4, 1)
    # pred = pred[target!=255, :].permute(1, 0).unsqueeze(0)
    # soft_label = soft_label[target!=255, :].permute(1, 0).unsqueeze(0)
    # kl_div = nn.KLDivLoss(reduction='none', log_target=True)
    criterion = nn.KLDivLoss(reduction='none')
    
    # student_loss = BCE_ssc_loss(pred, target, class_weights, 0.54)
    # loss = alpha * student_loss + (1-alpha) * dist_loss
    # pred = pred.permute(1, 0)
    # soft_label = soft_label.permute(1, 0)
    # criterion = nn.CrossEntropyLoss(
    #     reduction="none"
    # )
    loss = criterion(F.log_softmax(pred, dim=1), F.softmax(soft_label, dim=1))
    loss_sum = torch.sum(loss, dim=1)
    loss_valid = loss_sum[target!=255]
    # alpha = 0.7
    # loss = loss * alpha
    # loss = criterion(torch.log(pred), soft_label)

    # kl_loss = torch.mean(kl_div(F.log_softmax(pred/T, dim=1), F.softmax(soft_label/T, dim=1)), dim=-1)
    # kl_loss = kl_div(pred, soft_label)
    # loss = torch.sum(loss, dim=1)
    # loss_valid = loss[target!=255]
    # loss_valid_mean = torch.mean(loss) * (T * T)
    loss_valid_mean = torch.mean(loss_valid) 
    return loss_valid_mean


def disillation_decoder_loss(pred, soft_label, target, T=10):
    kl_div = nn.KLDivLoss(reduction='batchmean')
    # kl_div = nn.KLDivLoss(reduction='none')
    
    # student_loss = BCE_ssc_loss(pred, target, class_weights, 0.54)
    # loss = alpha * student_loss + (1-alpha) * dist_loss
    kl_loss = kl_div(F.log_softmax(pred/T, dim=1), F.softmax(soft_label/T, dim=1))
    # loss_valid = kl_loss[target!=255]
    loss_valid_mean = kl_loss * (T * T)
    return loss_valid_mean

# from monai.metrics import DiceMetric
import torch
from torchmetrics import MeanMetric
import segmentation_models_pytorch_3d as smp
from utils import cal_diceMetric_new, save_pred_nii_simple
from tqdm import tqdm
import numpy as np
from config import *
# from utils import save_pred

def train_one_epoch(
    opt,
    model,
    loader,
    optimizer,
    criterion,
    num_classes,
    device="cpu",
    epoch_idx=0,
    total_epochs=50,
    scheduler=None,
):
    model.train()

    loss_dice_record, loss_focus_record = MeanMetric(), MeanMetric()
    metric_record = MeanMetric()
    dice_record = [MeanMetric() for _ in range(4)]

    loader_len = len(loader)

    # with tqdm(total=loader_len, ncols=120) as tq:
        # tq.set_description(f"Train :: Epoch {epoch_idx}/{total_epochs}")
    for iter_, (data, target, mask_affines, save_path_nii, mask_crop_idx, tp_names) in enumerate(loader):
        # tq.update(1)
        # if iter_ >=4:
        #     break
        data, target = data.to(device).float(), target.to(device).long()
        print(f"input data.shape: {data.shape}")
        print(f"input mask.shape: {target.shape}")
        # print(f"data.shape: {data.shape}") ## [B, C, D, H, W]
        # print(f"target.shape: {target.shape}") ## [B, D, H, W]
        optimizer.zero_grad()

        output_dict = model(data)
        # print(f"output logits.shape: {output_dict.shape}") ## [B, C, D*2, H, W]

        # target = target.argmax(dim=1)  # one-hot to class index

        clsfy_out = output_dict
        if opt.mod == 'long':
            tp1_pred = clsfy_out[0]
            tp2_pred = clsfy_out[1]
            tp1_target = target[:,0,::]
            tp2_target = target[:,1,::]
            loss_dice1, loss_focus1 = criterion(tp1_pred, tp1_target)
            loss_dice2, loss_focus2 = criterion(tp2_pred, tp2_target)
            loss_dice = (loss_dice1 + loss_dice2)/2
            loss_focus = (loss_focus1 + loss_focus2)/2
        else:
            loss_dice, loss_focus = criterion(clsfy_out, target)
        loss = 0.5*loss_dice + 0.5*loss_focus
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if (epoch_idx % opt.save_nii_freq == 0 and epoch_idx != 0) or (epoch_idx == total_epochs - 1):
            # if True:
                if opt.if_save_nii:
                    save_pred_nii_simple(clsfy_out, epoch_idx, save_path_nii, mask_affines, mask_crop_idx, mod=opt.mod)


            if opt.mod == 'long':
                tp1_pred_idx = tp1_pred.argmax(dim=1)
                tp2_pred_idx = tp2_pred.argmax(dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    tp1_pred_idx, tp1_target, mode='multiclass', num_classes=num_classes
                )
                metric_macro1 = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
                tp, fp, fn, tn = smp.metrics.get_stats(
                    tp2_pred_idx, tp2_target, mode='multiclass', num_classes=num_classes
                )
                metric_macro2 = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
                metric_macro = (metric_macro1 + metric_macro2) / 2
            else:
                pred_idx = clsfy_out.argmax(dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_idx, target, mode='multiclass', num_classes=num_classes
                )
                metric_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            if opt.mod == 'long':
                # tp1_pred = clsfy_out[0]
                # tp2_pred = clsfy_out[1]
                # tp1_target = target[:,0,::]
                # tp2_target = target[:,1,::]
                dice_score1 = cal_diceMetric_new(tp1_pred, tp1_target, 4, mod=opt.mod)
                dice_score2 = cal_diceMetric_new(tp2_pred, tp2_target, 4, mod=opt.mod)
                dice_score_0 = (dice_score1 + dice_score2) / 2
            else:
                dice_score_0 = cal_diceMetric_new(clsfy_out, target, 4, mod=opt.mod)

            for i in range(4):
                dice_record[i].update(dice_score_0[i], weight=data.shape[0])
            loss_dice_record.update(loss_dice.detach().cpu(), weight=data.shape[0])
            loss_focus_record.update(loss_focus.detach().cpu(), weight=data.shape[0])
            # loss_record.update(loss.detach().cpu(), weight=data.shape[0])
            metric_record.update(metric_macro.cpu(), weight=data.shape[0])

            # ✅ 替代 tqdm.set_postfix_str —— 使用 tqdm.write() 每步清晰打印日志
            tqdm.write(
                f"[Epoch {epoch_idx:>3}/{total_epochs}] "
                f"Iter {iter_ + 1:>3}/{loader_len} | "
                f"IoU: {metric_macro:.4f} | "
                f"Dice: [{', '.join(f'{d:.4f}' for d in dice_score_0)}] | "
                f"loss_dice: {loss_dice:.4f} | "
                f"loss_focus: {loss_focus:.4f} | "
            )

    scheduler.step()
    # epoch_loss = {}
    loss_dice = loss_dice_record.compute()
    loss_focus = loss_focus_record.compute()
    epoch_metric = metric_record.compute()
    epoch_dice = [m.compute().item() for m in dice_record]


    return loss_dice, loss_focus, epoch_metric, epoch_dice



# Validation function, logging macro IoU and per-class IoU.
def validate(
    opt,
    model,
    loader,
    criterion,
    device,
    num_classes,
    epoch_idx,
    total_epochs
):
    model.eval()
    
    
    loss_dice_record, loss_focus_record = MeanMetric(), MeanMetric()
    metric_record = MeanMetric()
    dice_record = [MeanMetric() for _ in range(4)]
 
    loader_len = len(loader)
    
    
    for iter_, (data, target, mask_affines, save_path_nii, mask_crop_idx, tp_names) in enumerate(loader):
        # if iter_ >=4:
        #     break
        data, target = data.to(device).float(), target.to(device).long()

        with torch.no_grad():
            output_dict = model(data)
            if (epoch_idx % opt.save_nii_freq == 0 and epoch_idx != 0) or (epoch_idx == total_epochs - 1):
            # if True:
                if opt.if_save_nii:
                    save_pred_nii_simple(output_dict, epoch_idx, save_path_nii, mask_affines, mask_crop_idx, mod=opt.mod)

        clsfy_out = output_dict
        if opt.mod == 'long':
            tp1_pred = clsfy_out[0]
            tp2_pred = clsfy_out[1]
            tp1_target = target[:,0,::]
            tp2_target = target[:,1,::]
            loss_dice1, loss_focus1 = criterion(tp1_pred, tp1_target)
            loss_dice2, loss_focus2 = criterion(tp2_pred, tp2_target)
            loss_dice = (loss_dice1 + loss_dice2)/2
            loss_focus = (loss_focus1 + loss_focus2)/2
        else:
            loss_dice, loss_focus = criterion(clsfy_out, target)

        if opt.mod == 'long':
            tp1_pred_idx = tp1_pred.argmax(dim=1)
            tp2_pred_idx = tp2_pred.argmax(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                tp1_pred_idx, tp1_target, mode='multiclass', num_classes=num_classes
            )
            metric_macro1 = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            tp, fp, fn, tn = smp.metrics.get_stats(
                tp2_pred_idx, tp2_target, mode='multiclass', num_classes=num_classes
            )
            metric_macro2 = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            metric_macro = (metric_macro1 + metric_macro2) / 2
        else:
            pred_idx = clsfy_out.argmax(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_idx, target, mode='multiclass', num_classes=num_classes
            )
            metric_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')

        # 计算 Dice
        if opt.mod == 'long':
            # tp1_pred = clsfy_out[0]
            # tp2_pred = clsfy_out[1]
            # tp1_target = target[:,0,::]
            # tp2_target = target[:,1,::]
            dice_score1 = cal_diceMetric_new(tp1_pred, tp1_target, 4, mod=opt.mod)
            dice_score2 = cal_diceMetric_new(tp2_pred, tp2_target, 4, mod=opt.mod)
            dice_score_0 = (dice_score1 + dice_score2) / 2
        else:
            dice_score_0 = cal_diceMetric_new(clsfy_out, target, 4, mod=opt.mod)
        for i in range(4):
            dice_record[i].update(dice_score_0[i], weight=data.shape[0])
        
        loss_dice_record.update(loss_dice.detach().cpu(), weight=data.shape[0])
        loss_focus_record.update(loss_focus.detach().cpu(), weight=data.shape[0])
        metric_record.update(metric_macro.cpu(), weight=data.shape[0]) #data.shape = batch

        
        # ✅ 替代 tqdm.set_postfix_str —— 使用 tqdm.write() 每步清晰打印日志
        tqdm.write(
            f"[Val Epoch {epoch_idx:>3}/{total_epochs}] "
            f"Iter {iter_ + 1:>3}/{loader_len} | "
            f"IoU: {metric_macro:.4f} | "
            f"Dice: [{', '.join(f'{d:.4f}' for d in dice_score_0)}] | "
            f"loss_dice: {loss_dice:.4f} | "
            f"loss_focus: {loss_focus:.4f} | "
        )
    val_loss_dice = loss_dice_record.compute()
    val_loss_focus = loss_focus_record.compute()
    valid_epoch_metric = metric_record.compute()
    valid_epoch_dice = [m.compute().item() for m in dice_record]
    
    return val_loss_dice, val_loss_focus, valid_epoch_metric, valid_epoch_dice


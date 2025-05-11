import os
import glob
import numpy as np
import matplotlib.pyplot as plt
 
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
# from torch.cuda import amp
 
# from torchmetrics.classification import MulticlassAccuracy
 
from torch.utils.data import DataLoader
import gc
 
import segmentation_models_pytorch_3d as smp
 

from loss import F_loss, myDiceLoss
from Unet import UNet3D
from train_setting import *
from config import *
from utils import *
from Brats_Dataset import *
import datetime

import torch, gc
gc.collect()
torch.cuda.empty_cache()

## initialize
seed_everything(42)
opt = get_config()
bs = opt.batch_size
OUTPUT_DIR = opt.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE, _ = get_default_device(gpu_id=opt.gpu_id)

## set logger and get config
current_time = datetime.datetime.now().strftime("%m%d_%H-%M")
path_log = os.path.join(OUTPUT_DIR, f"log_{current_time}") ## output log path
if not os.path.exists(path_log):
    os.mkdir(path_log)
CKPT_DIR = f"{path_log}/ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
logger = set_logging(path_log)
logging_recording(logger, opt)


# Initialize datasets with normalization only
train_dir = os.path.join(opt.DatasetDir, "train")
val_dir = os.path.join(opt.DatasetDir, "val")
train_dataset = BratsDataset(opt, path_log, train_dir, normalization=True, isTrain=True, mod=opt.mod)
val_dataset = BratsDataset(opt, path_log, val_dir, normalization=True, isTrain=False, mod=opt.mod)
train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers = opt.num_workers, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False, num_workers = opt.num_workers, collate_fn=custom_collate_fn)


# Print dataset statistics
logging.info(f"Total Training Samples: {len(train_dataset)}")
logging.info(f"Total Val Samples: {len(val_dataset)}")


## initialize model
F_LOSS = F_loss()
model = UNet3D(opt, 8 if opt.mod=='long' else 4, 8 if opt.mod=='long' else 4, device=DEVICE)
model_info_recording(logger, model)
optimizer = torch.optim.AdamW(
   model.parameters(),
   lr=opt.lr, 
   weight_decay=1e-2,# Regularization to avoid overfitting
   amsgrad=True  # Optional AMSGrad variant
)

sche_cos = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr*0.01)  # 最小学习率调高一点

if opt.if_load_model:
   print("Loading model and optimizer state...")
   # checkpoint = torch.load(opt.model_ckpt_path, map_location='cpu')
   checkpoint = torch.load(opt.model_ckpt_path, map_location=DEVICE, weights_only=True)
   model.load_state_dict(checkpoint, strict=False)
   # optimizer.load_state_dict(checkpoint["opt"])
   print("Model state loaded.")

# Move model to the correct device before the loop starts
model.to(DEVICE)
F_LOSS.to(DEVICE)


best_loss = float("inf")
best_dice_fore = 0.

output_dict = initialize_output_dict()

total_epochs = opt.epochs
if_recorded = False
for epoch in range(total_epochs):
   current_epoch = epoch + 1

   # Train one epoch
   
   train_loss_dice, train_loss_focus, train_IoU, train_dice = train_one_epoch(
      opt,
      model=model,
      loader=train_loader,
      optimizer=optimizer,
      criterion=F_LOSS,
      num_classes=4,
      device=DEVICE,
      epoch_idx=current_epoch,
      total_epochs=total_epochs,
      scheduler=sche_cos,
   )

   dice_str = [f"{d:.4f}" for d in train_dice]
   logging.info(f"Epoch: {current_epoch}, IoU: {train_IoU:.4f}, Dice: {dice_str} && [{np.mean(train_dice):.4f}, {np.mean(train_dice[1:]):.4f}]")

   output_dict['train']['loss']['dice'].append(train_loss_dice)
   output_dict['train']['loss']['focus'].append(train_loss_focus)
   output_dict['train']['loss']['total'].append(train_loss_dice + train_loss_focus)

   output_dict['train']['IoU'].append(train_IoU)
   output_dict['train']['Dice'].append(train_dice)
   train_IoU_arr = np.array(output_dict['train']['IoU'])
   train_dice_arr = np.array(output_dict['train']['Dice'])
   save_all_plot(output_dict['train']['loss'], train_IoU_arr, train_dice_arr, path_log, isTrain=True)
   # Validate after each epoch
   with torch.no_grad():
      valid_loss_dice, valid_loss_focus, valid_IoU, valid_dice = validate(
            opt,
            model=model,
            loader=val_loader,
            criterion=F_LOSS,
            device=DEVICE,
            num_classes=4,
            epoch_idx=current_epoch,
            total_epochs=total_epochs,
      )
   dice_str_val = [f"{d:.4f}" for d in valid_dice]
   logging.info(f"[Val] Epoch: {current_epoch}, IoU: {valid_IoU:.4f}, Dice: {dice_str_val} && [{np.mean(valid_dice):.4f}, {np.mean(valid_dice[1:]):.4f}]")

   output_dict['val']['loss']['dice'].append(valid_loss_dice)
   output_dict['val']['loss']['focus'].append(valid_loss_focus)
   total_loss = valid_loss_dice + valid_loss_focus
   output_dict['val']['loss']['total'].append(total_loss)
   output_dict['val']['IoU'].append(valid_IoU)
   output_dict['val']['Dice'].append(valid_dice)
   # valid_loss_arr = np.array(output_dict['val']['loss'])
   valid_IoU_arr = np.array(output_dict['val']['IoU'])
   valid_dice_arr = np.array(output_dict['val']['Dice'])
   save_all_plot(output_dict['val']['loss'], valid_IoU_arr, valid_dice_arr, path_log, isTrain=False)

   # Step the Cosine Annealing LR scheduler
   # scheduler.step()

   # Save the model if validation loss improves
   if np.mean(valid_dice[1:]) > best_dice_fore:
      best_dice_fore = np.mean(valid_dice[1:])
      torch.save(model.state_dict(), os.path.join(CKPT_DIR, "ckpt.pth"))
      save_model_ckpt(model, os.path.join(CKPT_DIR, "ckpt.pth"))
      logging.info(f"Model improved and saved at epoch {current_epoch}")
   
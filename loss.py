import torch.nn as nn
import segmentation_models_pytorch_3d as smp
import torch
import numpy as np
import torch.nn.functional as F

class F_loss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",          # For multi-class segmentation
            classes=None,               # Compute the loss for all classes
            log_loss=False,             # Do not use log version of Dice loss
            from_logits=True,           # Model outputs are raw logits
            smooth=1e-5,                # A small smoothing factor for stability
            ignore_index=None,          # Don't ignore any classes
            eps=1e-7                    # Epsilon for numerical stability
        )

        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass",          # Multi-class segmentation
            alpha=0.25,                 # class weighting to deal with class imbalance
            gamma=2.0                   # Focusing parameter for hard-to-classify examples
        )

    def forward(self, output, target):
        # B, C, D, H, W = output.shape

        # dice_loss = 0.0
        # for d in range(D):
        #     output_slice = output[:, :, d, :, :]  # [B, C, H, W]
        #     target_slice = target[:, d, :, :]
        #     dice_loss += self.dice_loss(output_slice, target_slice)
        # loss1_d = dice_loss / D

        # dice_loss = 0.0
        # for h in range(H):
        #     output_slice = output[:, :, :, h, :].contiguous()  # Ensure contiguous memory layout
        #     target_slice = target[:, :, h, :].contiguous()  # Ensure contiguous memory layout
        #     dice_loss += self.dice_loss(output_slice, target_slice)
        # loss1_h = dice_loss / H

        # dice_loss = 0.0
        # for w in range(W):
        #     output_slice = output[:, :, :, :, w]
        #     target_slice = target[:, :, :, w]
        #     dice_loss += self.dice_loss(output_slice, target_slice)
        # loss1_w = dice_loss / W

        # loss1 = (loss1_d + loss1_h + loss1_w) / 3.0

        print(f'Output shape: {output.shape}, Target shape: {target.shape}')
        # print("Output unique values:", torch.unique(output))

        loss1 = self.dice_loss(output, target)
        loss2 = self.focal_loss(output, target)
        return loss1, loss2



class myBinaryDice(nn.Module):

    def __init__(self, smooth=1e-8):
        super(myBinaryDice, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        '''
        calculate bi-class dice loss for 3D-UNet
        Args:
            predict: [B, D, H, W], unique=[0, 1]
            target: [B, D, H, W], unique=[0, 1]

        Returns:
            class_num: 4
            dice_loss: single torch.Tensor
        '''
        assert predict.shape == target.shape, f'In binary dice calculation, predict {predict.shape} & target {target.shape} do not match'
        predict_flat = predict.view(predict.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + self.smooth

        # loss = 1 - 2 * num / den
        # loss = torch.sum(loss)

        
        intersection = (predict_flat * target_flat).sum(dim=1)
        union = predict_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        return dice_score
    

class myDiceLoss(nn.Module):

    def __init__(self, num_class, smooth=1e-8, weights=None):
        '''
        Args:
            smooth: float, smooth value to avoid division by zero
            weights: list, weights for different classes
            labels: list, unique labels in target
        '''
        super(myDiceLoss, self).__init__()
        self.smooth = smooth
        if weights is None:
            self.weights = [1., 8., 3., 3.]
        else:
            self.weights = weights
        print(f'Weights: {self.weights}')
        self.classes = num_class
        self.labels = list(range(self.classes))

        # assert len(weights) == len(labels), 'Not matched weights & labels lengths'

    def forward(self, predict, target):
        '''
        calculate multi-class dice loss for 3D-UNet
        Args:
            predict: [B, C, D, H, W], logits
            target: [B, D, H, W]

        Returns:
            dice_loss: single torch.Tensor
        '''
        unique, counts  = target.unique(return_counts=True) # 获取 target 中的 unique labels
        # print(f"unique: {unique}; counts: {counts}")
        unique_list = unique.tolist()

        # 归一化权重
        weights = torch.tensor(self.weights)
        weights /= weights.sum()

        # print(f"weights: {weights}")
        predict = torch.softmax(predict, dim=1) # [B, C, D, H, W] -> [B, C, D, H, W]


        ## one-hot encode target
        target = target.unsqueeze(1)
        target = torch.zeros_like(predict).scatter_(1, target, 1) # -> [B, C, D, H, W]

        loss_list = []
        # dice_score_list = []
        ## calculate dice loss for each class
        for i in range(len(self.labels)):

            dice_score = myBinaryDice(smooth=self.smooth)(predict[:, i], target[:, i])
            dice_score = torch.mean(dice_score)
            # dice_score_list.append(dice_score)
            loss_list.append((1-dice_score)*weights[i])
        
        return sum(loss_list)



class Pairwise_Feature_Similarity(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        # self.l2norm = Normalize(2)

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        print('Initialized pairwise similarity loss')

    def pairwise_cosine_simililarity(self, x, y):
        assert x.size() == y.size(), f'wrong shape {x.size()} and {y.size()}'
        v = self._cosine_similarity(x,y) #(x.unsqueeze(1), y.unsqueeze(2))
       
        return v

    def forward(self, feat_A, feat_B):
        '''
        :param feats:  bs*2, num_patches, nc_feature
        :param labels_age: bs*2 -> currently not consider age
        :param labels_subjid: bs*2
        :return: loss range in [-1,1]
        '''
        # bs2, num_patches, nc = features.size()

        # bs = bs2//2
        # feat_A = features[:bs, ...].view(bs*num_patches, nc)
        # feat_B = features[bs:, ...].view(bs*num_patches, nc)

        tot = feat_A.shape[0]
        sim_pos = self.pairwise_cosine_simililarity(feat_A, feat_B)
        loss = -(sim_pos).sum() / (tot)  # negative cosine similarity
        return loss
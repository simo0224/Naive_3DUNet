import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from utils import crop_to_multiple_of_16, save_nii_simple
import SimpleITK as sitk
import glob

def center_crop_3d(image, crop_size):
    """对3D图像在XYZ轴上进行中心裁剪"""
    assert image.ndim == 3 or image.ndim == 4, "必须是3D或带通道的4D图像"
    
    if image.ndim == 4:
        c, x, y, z = image.shape
        cropped = image[:, 
                        x//2 - crop_size[0]//2 : x//2 + crop_size[0]//2,
                        y//2 - crop_size[1]//2 : y//2 + crop_size[1]//2,
                        z//2 - crop_size[2]//2 : z//2 + crop_size[2]//2]
    else:
        x, y, z = image.shape
        cropped = image[
                        x//2 - crop_size[0]//2 : x//2 + crop_size[0]//2,
                        y//2 - crop_size[1]//2 : y//2 + crop_size[1]//2,
                        z//2 - crop_size[2]//2 : z//2 + crop_size[2]//2]
    return cropped

class BratsDataset(Dataset):
    def __init__(self, opt, path_log, data_dir, normalization=True, isTrain=True, mod='single'):
        super().__init__()
        self.opt = opt
        self.isTrain = isTrain
        self.path_log = path_log
        self.mod = mod
        if mod == 'single':
            # self.img_list = sorted(glob.glob(os.path.join(data_dir, "**", "CT1_image_rig.nii.gz"), recursive=True))
            # self.mask_list = sorted(glob.glob(os.path.join(data_dir, "**", "ct1_seg_mask_rig.nii.gz"), recursive=True))
            self.img_list = sorted(glob.glob(os.path.join(data_dir, "**", "GED4.nii.gz"), recursive=True))
            self.mask_list = sorted(glob.glob(os.path.join(data_dir, "**", "mask_GED4.nii.gz"), recursive=True))
            self.img_list = self.img_list[1::2]
            self.mask_list = self.mask_list[1::2]
        elif mod == 'long':
            self.data_dir = data_dir
            self.pat_list = sorted(os.listdir(self.data_dir))
            self.pair_img_list = sorted(
                [sorted([os.path.join(self.data_dir, pat, week, "CT1_image_rig.nii.gz") for week in os.listdir(os.path.join(self.data_dir, pat))]) 
                 for pat in self.pat_list]
            )
            self.mask_dir = os.path.join(data_dir, "labels")
            self.pair_mask_list = sorted(
                [sorted([os.path.join(self.data_dir, pat, week, "ct1_seg_mask_rig.nii.gz") for week in os.listdir(os.path.join(self.data_dir, pat))]) 
                 for pat in self.pat_list]
            )
            assert len(self.pair_img_list) == len(self.pair_mask_list)
        else:
            raise ValueError("mod must be 'single' or 'long'")
        
        self.normalization = normalization
 
        # If normalization is True, set up a normalization transform
        if self.normalization:
            self.normalizer = transforms.Normalize(
                mean=[0.5], std=[0.5]
            )  # Adjust mean and std based on your data
 
    def load_file(self, filepath, if_image=True):
        # Load image or mask
        if if_image:
            temp_image_t1ce, img_affine = self.get_nii(filepath)
            scaler = MinMaxScaler()
            temp_image_t1ce = scaler.fit_transform(
                temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])
            ).reshape(temp_image_t1ce.shape)

            total_size = np.prod(temp_image_t1ce.shape)
            if (total_size > 1e7):
                temp_image_t1ce = F.interpolate(
                    torch.tensor(temp_image_t1ce).unsqueeze(0).unsqueeze(0), 
                    scale_factor = 0.5, 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze().numpy()
            # crop_size = tuple(s // 2 for s in temp_image_t1ce.shape)
            # temp_image_t1ce = center_crop_3d(temp_image_t1ce, crop_size)

            # temp_combined_images = np.expand_dims(temp_image_t1ce, 0)
            return temp_image_t1ce, img_affine
        else:
            temp_mask, mask_affine = self.get_nii(filepath)
            temp_mask = np.expand_dims(temp_mask, 0)

            # b, x, y, z = temp_mask.shape
            # crop_size = (x // 2, y // 2, z // 2)
            # temp_mask = center_crop_3d(temp_mask, crop_size)

            return temp_mask, mask_affine

    def get_nii(self, input_filename):
        image = sitk.ReadImage(input_filename)
        affine = {}
        affine['Spacing'] = image.GetSpacing()
        affine['Origin'] = image.GetOrigin()
        affine['Direction'] = image.GetDirection()
        image_data = sitk.GetArrayFromImage(image)
        return image_data, affine
    
    def get_4_modality(self, input_filename):
        ## CT1
        CT1_path = input_filename
        CT1_data, affine_CT1 = self.load_file(CT1_path, if_image=True)

        ## FLAIR, T1, T2
        base_dir_loc = os.path.dirname(input_filename)
        FLAIR_path = os.path.join(base_dir_loc, "FLAIR_image_rig.nii.gz")
        T1_path = os.path.join(base_dir_loc, "T1_image_rig.nii.gz")
        T2_path = os.path.join(base_dir_loc, "T2_image_rig.nii.gz")
        FLAIR_data, _ = self.load_file(FLAIR_path, if_image=True)
        T1_data, _ = self.load_file(T1_path, if_image=True)
        T2_data, _ = self.load_file(T2_path, if_image=True)

        temp_combined_images = np.stack([CT1_data, FLAIR_data, T1_data, T2_data], axis=3)
        combined_images = temp_combined_images.transpose(3, 0, 1, 2)
        return combined_images, affine_CT1
 
    def __len__(self):
        if self.mod=="single":
            return len(self.img_list)
        else:
            return len(self.pair_img_list)
    
    def __getitem__(self, idx):
        '''
            |--- Patient-001
                |--- week-000-1
                    |--- CT1_image_rig.mat
                    |--- CT1_image_rig.nii.gz
                    |--- ct1_seg_mask_rig.nii.gz
                    |--- FLAIR_image_rig.nii.gz
                    |--- _.mat
                    |--- T1_image_rig.nii.gz
                    └── T2_image_rig.nii.gz
                |---week-000-2
                    |--- CT1_image_rig.mat
                    |--- CT1_image_rig.nii.gz
                    |--- ct1_seg_mask_rig.nii.gz
                    |--- FLAIR_image_rig.nii.gz
                    |--- _.mat
                    |--- T1_image_rig.nii.gz
                    └── T2_image_rig.nii.gz
            |--- Patient-002
                |--- week-003
        
        '''
        if self.mod=='single':
            image_path = self.img_list[idx]
            mask_path = self.mask_list[idx]
            # Load the image and mask
            image, image_affine = self.load_file(image_path, if_image=True) ## [1, D, H, W]
            # mask, mask_affine = self.load_file(mask_path, if_image=False) ## [1, D, H, W]
            # image, im_crop_idx = crop_to_multiple_of_16(image)
            im_crop_idx = [0,0,0]
            tp_name = self.mask_list[idx][:-11]
            mask, mask_affine = self.load_file(mask_path, if_image=False) ## [1, D, H, W]
            # mask, mask_crop_idx = crop_to_multiple_of_16(mask)
            mask_crop_idx = [0,0,0]
            
        
            image = torch.from_numpy(image).unsqueeze(0) # Shape: C, D, H, W
            mask = torch.from_numpy(mask).squeeze()  # Shape: D, H, W

            sub_folder = 'train' if self.isTrain else 'val'
            pat_name = image_path.split("/")[-3]
            week_name = image_path.split("/")[-2]
            save_path_nii = os.path.join(self.path_log,  'niiData', sub_folder, pat_name, week_name)
            os.makedirs(save_path_nii, exist_ok=True)
            if self.opt.if_save_nii:
                save_nii_simple(image, save_path_nii, isMask=False, affine=image_affine, crop_offsets=im_crop_idx)
                save_nii_simple(mask, save_path_nii, isMask=True, affine=mask_affine, crop_offsets=mask_crop_idx)
            
            # Normalize the image if normalization is enabled
            if self.normalization:
                image = self.normalizer(image)
            
            return image, mask, mask_affine, save_path_nii, mask_crop_idx, tp_name
        else:
            #### get the neighbored data
            tp_img_dir = self.pair_img_list[idx]
            tp_mask_dir = self.pair_mask_list[idx]
            # tp_pair_name = tp_img_dir.split("/")[-1]
            tp_pair_name = tp_img_dir[0].split("/")[-3]
            # print(tp_pair_name)
            image1_path = tp_img_dir[0]
            image2_path = tp_img_dir[1]
            mask1_path = tp_mask_dir[0]
            mask2_path = tp_mask_dir[1]

            image1, image1_affine = self.get_4_modality(image1_path)
            image2, image2_affine = self.get_4_modality(image2_path)
            mask1, mask1_affine = self.load_file(mask1_path, if_image=False)
            mask2, mask2_affine = self.load_file(mask2_path, if_image=False)
            image1, im1_crop_idx = crop_to_multiple_of_16(image1)
            mask1, mask1_crop_idx = crop_to_multiple_of_16(mask1)
            image2, im2_crop_idx = crop_to_multiple_of_16(image2)
            mask2, mask2_crop_idx = crop_to_multiple_of_16(mask2)

            sub_folder = 'train' if self.isTrain else 'val'
            save_path_nii1 = os.path.join(self.path_log,  'niiData', sub_folder, tp_pair_name, "tp1")
            save_path_nii2 = os.path.join(self.path_log,  'niiData', sub_folder, tp_pair_name, "tp2")
            image1, image2 = torch.from_numpy(image1), torch.from_numpy(image2)
            mask1, mask2 = torch.from_numpy(mask1).squeeze(), torch.from_numpy(mask2).squeeze()  # Shape: D, H, W

            if self.opt.if_save_nii:
                os.makedirs(save_path_nii1, exist_ok=True)
                os.makedirs(save_path_nii2, exist_ok=True)
                save_nii_simple(image1[0], save_path_nii1, isMask=False, affine=image1_affine, crop_offsets=im1_crop_idx)
                save_nii_simple(mask1, save_path_nii1, isMask=True, affine=mask1_affine, crop_offsets=mask1_crop_idx)
                save_nii_simple(image2[0], save_path_nii2, isMask=False, affine=image2_affine, crop_offsets=im2_crop_idx)
                save_nii_simple(mask2, save_path_nii2, isMask=True, affine=mask2_affine, crop_offsets=mask2_crop_idx)
            
            if self.normalization:
                image1 = self.normalizer(image1)
                image2 = self.normalizer(image2)
            # images = torch.cat([image1, image2], dim=1) ## 在 D 维度上 cat
            # masks = torch.cat([mask1, mask2], dim=0) ## 在 D 维度上 cat
            images = torch.cat([image1, image2], dim=0) ## 在 C 维度上 cat
            # masks = torch.cat([mask1, mask2], dim=0) ## 在 C 维度上 cat
            masks = torch.stack([mask1, mask2], dim=0)  # Shape: 2, D, H, W
            assert images.shape[0] == 8, "concate from dim=C not right"
            mask_affines = [mask1_affine, mask2_affine]
            save_path_nii = [save_path_nii1, save_path_nii2]
            mask_crop_idx = [mask1_crop_idx, mask2_crop_idx]
            tp_name = ["_", "_"]

            return images, masks, mask_affines, save_path_nii, mask_crop_idx, tp_name


    
def custom_collate_fn(batch):
    """
    batch: List of tuples (image_choose, mask_choose, affine_mask2, save_path_nii, crop_offset_mask, tp_name)
    
    按照返回值的顺序组织 batch，并确保 batch 形式正确：
    - `image_choose` 和 `mask_choose` 会被 `stack` 成 Tensor
    - `affine_mask2`, `save_path_nii`, `crop_offset_mask`, `tp_name` 保持原格式，变成 list
    """
    image_choose_batch = torch.stack([item[0] for item in batch], dim=0)
    mask_choose_batch = torch.stack([item[1] for item in batch], dim=0)
    
    affine_mask2_batch = [item[2] for item in batch]  # 直接变成 list
    save_path_nii_batch = [item[3] for item in batch]
    crop_offset_mask_batch = [item[4] for item in batch]
    tp_name_batch = [item[5] for item in batch]
    
    return (image_choose_batch, mask_choose_batch, affine_mask2_batch, 
            save_path_nii_batch, crop_offset_mask_batch, tp_name_batch)


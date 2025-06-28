import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk

def cal_diceMetric_new(out, gt, num_class, mod='single'):
    '''
    使用 numpy 计算多类别 Dice 系数
    Arg: 
        out: (B, C=num_class, D, H, W) - 模型的输出 (torch.Tensor)
        gt: (B, D, H, W) - 真实标签掩码 (torch.Tensor)
        num_class: int - 类别数
    Return:
        dice_scores: (num_class,) - 每个类别的 Dice 系数
    '''
    # tensor to numpy
    with torch.no_grad():
        out_logits = out.cpu().numpy()
        mask = gt.cpu().numpy()
    pred_mask = np.argmax(out_logits, axis=1)  # [B, C, D, H, W] -> [B, D, H, W]

    dice_scores = np.zeros(num_class)

    for b in range(pred_mask.shape[0]): 
        pred_sample = pred_mask[b]
        mask_sample = mask[b]  # [B=b, C=1, D, H, W] -> [D, H, W]

        for cls in range(num_class):
            pred_i = (pred_sample == cls).astype(float)
            mask_i = (mask_sample == cls).astype(float)

            # 计算交集和 Dice 系数
            inter_area = np.sum(pred_i * mask_i)
            den = np.sum(pred_i) + np.sum(mask_i) + 1e-8
            num = 2 * inter_area + 1e-8
            dice_i = num / den

            dice_scores[cls] += dice_i

    # average Dice score
    dice_scores /= pred_mask.shape[0] 
    return dice_scores



def seed_everything(SEED):
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed_all(SEED)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False



def get_default_device(gpu_id):
   gpu_available = torch.cuda.is_available()
   return torch.device(f'cuda:{gpu_id}' if gpu_available else 'cpu'), gpu_available


def set_logging(path_file):
    # 获取当前时间，并格式化为适合文件名的字符串
    log_filename = os.path.join(path_file, "logging.txt")

    # 创建一个logger，如果已经存在处理器则不重复添加
    logger = logging.getLogger()

    # 检查是否已经有文件处理器，如果没有再添加
    if not logger.handlers:  # 确保处理器不重复添加
        logger.setLevel(logging.INFO)  # 设置全局日志级别

        # 创建一个文件处理器，将日志写入文件
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # 创建一个控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 定义日志格式
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到 logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def logging_recording(logger, opt):
    """
    Logs each configuration option and its value from the given `opt` object.

    Parameters:
    - logger: A configured logger to write logs.
    - opt: An object containing the parsed configuration options.
    """
    logger.info("---------------Training Configuration-------------")
    # 获取所有键的最大长度
    max_key_len = max(len(key) for key in vars(opt))
    
    # 遍历 opt 的所有属性，并记录每个参数的名称和值，确保 ":" 对齐
    for key, value in vars(opt).items():
        # 使用 ljust 格式化，使得所有的 ":" 对齐
        logger.info(f"{key.ljust(max_key_len)} : {value}")

def initialize_output_dict():
    output_dict = {}
    output_dict['train'] = {}
    output_dict['val'] = {}
    output_dict['train']['loss'] = {}
    output_dict['train']['loss']['dice'] = []
    output_dict['train']['loss']['focus'] = []
    output_dict['train']['loss']['sim'] = []
    output_dict['train']['loss']['total'] = []
    output_dict['train']['IoU'] = []
    output_dict['train']['Dice'] = []
    output_dict['val']['loss'] = {}
    output_dict['val']['loss']['dice'] = []
    output_dict['val']['loss']['focus'] = []
    output_dict['val']['loss']['sim'] = []
    output_dict['val']['loss']['total'] = []
    output_dict['val']['IoU'] = []
    output_dict['val']['Dice'] = []
    return output_dict


def model_info_recording(logger, model):
    logger.info(f"-------------------TRAINING MODEL------------------")
    logger.info(model)
    logger.info(f"------------------------Over----------------------")

def save_dice_plot(data, output_dir, isTrain=True):
    '''
        data.shape = [epoch_num, class_num]    
    '''

    # x 轴是 epoch
    epochs = np.arange(data.shape[0])

    # 绘图
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.plot(epochs, data[:, i], label=f'Class {i}')

    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.ylim(0, 1)

    if isTrain:
        title = 'Train Dice'
        save_name = os.path.join(output_dir, 'Train_dice.png')
    else:
        title = 'Val Dice'
        save_name = os.path.join(output_dir, 'Val_dice.png')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_name, dpi=300)
    plt.close()


def save_IoU_plot(data, output_dir, isTrain=True):
    '''
        data.shape = [epoch_num, class_num]    
    '''

    # x 轴是 epoch
    epochs = np.arange(data.shape[0])

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data)

    plt.xlabel('Epoch')
    plt.ylabel('IoU')

    if isTrain:
        title = 'Train IoU'
        save_name = os.path.join(output_dir, 'Train_IoU.png')
    else:
        title = 'Val IoU'
        save_name = os.path.join(output_dir, 'Val_IoU.png')
    plt.title(title)
    # plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_name, dpi=300)
    plt.close()


def save_loss_plot(data, output_dir, isTrain=True, loss_name='loss'):
    '''
        data.shape = [epoch_num, class_num]    
    '''

    # x 轴是 epoch
    epochs = np.arange(data.shape[0])

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if isTrain:
        title = f'Train {loss_name}'
        save_name = os.path.join(output_dir, f'Train_{loss_name}.png')
    else:
        title = 'Val {loss_name}'
        save_name = os.path.join(output_dir, f'Val_{loss_name}.png')
    plt.title(title)
    # plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_name, dpi=300)
    plt.close()

def save_all_plot(loss_list, IoU_data, Dice_data, path_log, isTrain = True):
    
    train_loss_dice_arr = np.array(loss_list['dice'])
    train_loss_focus_arr = np.array(loss_list['focus'])
    # train_loss_focus_arr = np.array(train_loss_focus)
    # train_loss_sim_arr = np.array(train_loss_sim)
    train_loss_tot_arr = np.array(loss_list['total'])
    save_loss_plot(train_loss_dice_arr, path_log, isTrain, loss_name='loss_dice')
    save_loss_plot(train_loss_focus_arr, path_log, isTrain, loss_name='loss_focus')
    save_loss_plot(train_loss_tot_arr, path_log, isTrain, loss_name='loss_total')
    save_IoU_plot(IoU_data, path_log, isTrain)
    save_dice_plot(Dice_data, path_log, isTrain)


# def save_pred(save_path, output_dict, mask_affines, tp_names, isTrain=True):
    
#     pred_logits_np = output_dict.detach().cpu().numpy()
#     pred_labels = np.argmax(pred_logits_np, axis=1)
#     output_dir = save_path
#     if isTrain:
#         output_dir = os.path.join(output_dir, "niigz", "train/")
#     else:
#         output_dir = os.path.join(output_dir, "niigz", "val/")
#     os.makedirs(output_dir, exist_ok=True)
#     for i in range(pred_labels.shape[0]):
#         pred_i = pred_labels[i]  # shape: [D, H, W]
#         sub_dir = os.path.join(output_dir, tp_names[i])
#         os.makedirs(sub_dir, exist_ok=True)
#         # 创建 Nifti 图像
#         nifti_img = nib.Nifti1Image(pred_i.astype(np.uint8), affine=mask_affines[i])
        
#         # 保存
#         save_path = os.path.join(sub_dir, f"pred_{tp_names[i]}.nii.gz")
#         nib.save(nifti_img, save_path)


# def save_ori(output_dir, mask, mask_affine, image, image_affine, tp_name, isTrain=True):
#     if isTrain:
#         output_dir = os.path.join(output_dir, "niigz", "train/")
#     else:
#         output_dir = os.path.join(output_dir, "niigz", "val/")
#     os.makedirs(output_dir, exist_ok=True)
#     sub_dir = os.path.join(output_dir, tp_name)
#     os.makedirs(sub_dir, exist_ok=True)
#     # 创建 Nifti 图像
#     nifti_mask = nib.Nifti1Image(mask.cpu().numpy().astype(np.uint8), affine=mask_affine)
#     nifti_img = nib.Nifti1Image(image, affine=image_affine)
    
#     # 保存
#     save_path_mask = os.path.join(sub_dir, f"mask.nii.gz")
#     save_path_image = os.path.join(sub_dir, f"image.nii.gz")
#     nib.save(nifti_mask, save_path_mask)
#     nib.save(nifti_img, save_path_image)


def load_image_resample(img_path, target_spacing=(1.0, 1.0, 1.0), interp=sitk.sitkLinear):
    """
    读取 NIfTI 图像，统一方向（LPS）并重采样 spacing，返回 numpy array。

    参数：
    - img_path: 图像路径
    - target_spacing: 目标 spacing（默认 1mm³）
    - interp: 插值方式（图像用 sitkLinear，mask 用 sitkNearestNeighbor）

    返回：
    - data: numpy array, shape=[Z, Y, X]
    """
    # 读取图像
    image = sitk.ReadImage(img_path)

    # 统一方向（LPS）
    image = sitk.DICOMOrient(image, 'LPS')

    # 获取原始参数
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # 计算新尺寸：new_size = old_size * (old_spacing / new_spacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    # 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp)

    image_resampled = resampler.Execute(image)

    # 设置新的原点
    affine = {}
    affine['Spacing'] = image_resampled.GetSpacing()
    affine['Origin'] = image_resampled.GetOrigin()
    affine['Direction'] = image_resampled.GetDirection()

    # 转为 numpy array（[Z, Y, X]）
    data = sitk.GetArrayFromImage(image_resampled)

    return data, affine



def crop_to_multiple_of_16(array, mod='single_seg'):
    """
    change the shape to fit UNet pool and upconv
    array:  (B, D, H, W)
    ouput:  (B, D', H', W')
    """
    original_shape = array.shape[1:]  # 忽略batch size
    # cropped_shape = get_closest_multiple_of_16(original_shape)
    # cropped_shape = tuple([160,160,160])
    if mod == 'long_seg':
        cropped_shape = tuple([160,160,160])
        # cropped_shape = tuple([144,160,160])
        # cropped_shape = tuple([144,144,144])
        # cropped_shape = tuple([96,128,128])
        # cropped_shape = tuple([96,96,96])
        # cropped_shape = tuple([64,64,64])
    elif mod == 'single_seg':
        # cropped_shape = tuple([128,128,128])
        cropped_shape = tuple([144,256,256])
        # cropped_shape = tuple([96,96,96])
    else:
        raise ValueError("mod should be 'long_seg' or 'single_seg'")
    # cropped_shape = tuple([144, 160, 160])
    # cropped_shape = tuple([96,96,96])

    ## calculate the difference between the original shape and the cropped shape
    diff = np.array(original_shape) - np.array(cropped_shape)
    crop_offsets = [diff[i] // 2 for i in range(3)]  # 每个维度的起始裁剪坐标
    
    D_interval = slice(crop_offsets[0], crop_offsets[0]+ cropped_shape[0])
    H_interval = slice(crop_offsets[1], crop_offsets[1]+ cropped_shape[1])
    W_interval = slice(crop_offsets[2], crop_offsets[2]+ cropped_shape[2])

    cropped_array = array[:, D_interval, H_interval, W_interval]
    
    return cropped_array, crop_offsets


def save_pred_nii_simple(pred, epoch, path_folder, affine=None, crop_offsets=None, mod='single'):

    if mod == 'single':
        for i in range(pred.shape[0]):
            # 1. 转换 logits 为分类标签 (假设 argmax 方式进行多分类)
            # 假设 pred 大小为 (B=1, C=1, D, H, W)，我们去掉 batch 维度和 channel 维度
            pred_i = pred[i]  # 得到 (C, D, H, W)
            pred_labels = torch.argmax(torch.softmax(pred_i, dim=0), dim=0)  # 对 C 维度执行 argmax，得到 (D, H, W)
            
            # 去掉 batch 维度，得到 (D, H, W)
            
            # 2. 将张量转换为 NumPy 数组，并确保它在 CPU 上
            pred_numpy = pred_labels.cpu().numpy().astype(np.int32)
            
            # 3. 创建保存路径，文件名为 "pred_{epoch}.nii.gz"
            file_name = f"pred_{epoch}.nii.gz"
            # file_name = f"pred.nii.gz"
            file_path = os.path.join(path_folder[i], file_name)
            
            affine_i = affine[i]
            crop_offsets_i = crop_offsets[i]
            save_nii(pred_numpy, affine_i, file_path, crop_offsets_i)
    else:
        '''
        pred: a list of 2 tp logits, each logit is (B, C=4, D, H, W)
        path_folder: a list of sublist, each sublist contains 2 tp folder_names
        '''
        assert pred.shape[0] == len(path_folder), "pred and path_folder should have the same length"

        # 3. 创建保存路径，文件名为 "pred_{epoch}.nii.gz"
        pred_numpy_j = torch.argmax(torch.softmax(pred, dim=1), dim=1)  # 对 C 维度执行 argmax，得到 (B, D, H, W)
        pred_numpy_j = pred_numpy_j.cpu().numpy().astype(np.int32)  # 转为 numpy 数组，并确保它在 CPU 上
        
        for i in range(pred_numpy_j.shape[0]):
            pred_j_i = pred_numpy_j[i]
            file_name = f"pred_{epoch}.nii.gz"
            file_path = os.path.join(path_folder[i][1], file_name)
            
            if affine is None:
                affine_j_i = np.eye(4)
            else:
                affine_j_i = affine[i][1]
            if crop_offsets:
                crop_offsets_i = crop_offsets[i][1]
            save_nii(pred_j_i, affine_j_i, file_path, crop_offsets_i)


def save_nii_simple(pred, path_folder, aug_type=None, isMask=False, affine=None, crop_offsets=None):
    
    # 得到 (D, H, W)
    pred_labels = pred.squeeze(0)
    
    pred_numpy = pred_labels.cpu().numpy() 
    
    if isMask:
        file_name = "mask.nii.gz"
    else:
        if aug_type:
            file_name = f"image_{aug_type}.nii.gz"
        else:
            file_name = f"image_None.nii.gz"
    file_path = os.path.join(path_folder, file_name)
    
    if affine is None:
        affine = np.eye(4) 
    
    save_nii(pred_numpy, affine, file_path, crop_offsets)

# def save_nii(image_data, affine, output_filename, crop_offsets=None):

#     # 创建新的图像
#     # affine = {}
#     # if type(affine0['Spacing']) is tuple:
#     #     affine = affine0
#     # else:
#     #     for key, _ in affine0.items():
#     #         affine[key] = [affine0[key][i].item() for i in range(len(affine0[key]))]

#     new_image = sitk.GetImageFromArray(image_data)
#     new_image.SetSpacing(affine['Spacing'])      # 设置间距
#     new_image.SetDirection(affine['Direction'])         # 设置方向（仿射矩阵）
#     if crop_offsets is None:
#         crop_offsets = [0, 0, 0]
#         new_origin = np.array(affine['Origin']) + np.array(crop_offsets) * np.array(affine['Spacing'])
#         new_image.SetOrigin(new_origin.tolist())        # 设置原点
#     else:
#         new_image.SetOrigin(affine['Origin'])        # 设置原点
#     # 保存为新的 .nii.gz 文件
#     sitk.WriteImage(new_image, output_filename)

def save_nii(pred_numpy, affine, file_path, crop_offsets=None):
    # 确保 affine['Direction'] 是一维列表且包含 9 个元素
    direction = affine['Direction']
    
    # 如果 direction 是一个 3x3 的矩阵（列表的列表），则展平它
    if isinstance(direction, list) and isinstance(direction[0], list):
        # 例如 direction 是 [[d00, d01, d02], [d10, d11, d12], [d20, d21, d22]]
        direction = [item for sublist in direction for item in sublist]

    # 确保 direction 是一个包含 9 个元素的列表
    if len(direction) != 9:
        raise ValueError(f"Invalid direction matrix, it should contain 9 elements but has {len(direction)}.")

    # 创建一个 SimpleITK 图像并设置其方向
    new_image = sitk.GetImageFromArray(pred_numpy)
    new_image.SetDirection(direction)  # 设置方向（仿射矩阵）
    
    # 设置其他属性：间距和原点
    new_image.SetSpacing(affine['Spacing'])
    new_image.SetOrigin(affine['Origin'])

    # 保存图像
    sitk.WriteImage(new_image, file_path)

def save_model_ckpt(model, path):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def Prediction_to_nii(output_dict, target_spacing, original_origin, original_direction):
    np_array = output_dict.cpu().numpy()  # Convert to numpy array

    if (np_array.ndim == 4):
        np_array = np.argmax(np_array, axis=0)  # Convert to class labels if needed
    elif (np_array.ndim == 3):
        np_array = np_array

    image = sitk.GetImageFromArray(np_array.astype(np.uint8))  # Convert to SimpleITK image
    image.SetSpacing(target_spacing)  # Set target spacing
    image.SetOrigin(original_origin)  # Set original origin
    image.SetDirection(original_direction)  # Set original direction
    return image  # Return the SimpleITK image

def resample_image(itk_image, out_spacing, out_shape, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    original_direction = itk_image.GetDirection()
    original_origin = itk_image.GetOrigin()

    out_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(itk_image.GetSize(), original_spacing, out_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_shape)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)  # Default interpolator
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        # resampler.SetInterpolator(sitk.sitkLabelGaussian)
        # resampler.SetSigma(3.0)   # 控制边界模糊的程度，数值越大越模糊（默认1.0）
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(itk_image)

def Resampling_back(output_dict, original_spacing, original_origin, original_direction, original_shape, padding_info):

    output_dict = output_dict.squeeze(0)  # Remove batch dimension if present
    
    original_origin = original_origin[0]  # Assuming original_origin is a list of lists
    original_direction = original_direction[0]  # Assuming original_direction is a list of lists
    original_spacing = original_spacing[0]  # Assuming original_spacing is a list of lists
    original_shape = original_shape[0]  # Assuming original_shape is a list of lists

    # Remove padding
    print(padding_info)
    [z0, z1]= padding_info[0][0]
    [y0, y1] = padding_info[0][1]
    [x0, x1] = padding_info[0][2]
    print(f"Before removing padding: {output_dict.shape}")
    if len(output_dict.shape) == 4:
        output_dict = output_dict[:, z0:len(output_dict[0]) - z1, y0:len(output_dict[0][0]) - y1, x0:len(output_dict[0][0][0]) - x1]
    else:
        output_dict = output_dict[z0:len(output_dict) - z1, y0:len(output_dict[0]) - y1, x0:len(output_dict[0][0]) - x1]

    print(f"Output shape after removing padding: {output_dict.shape}")
    original_origin = list(map(float, original_origin))  # Convert to float
    original_direction = list(map(float, original_direction))  # Convert to float
    original_spacing = list(map(float, original_spacing))  # Convert to float
    # Prediction to nii
    pred_nii = Prediction_to_nii(output_dict, (1.0, 1.3, 1.3), original_origin, original_direction)

    # Resample to original spacing
    # original_spacing = list(map(float, original_spacing))
    # original_shape = list(map(int, original_shape))

    # resampler = sitk.ResampleImageFilter()
    # resampler.SetOutputSpacing(original_spacing)
    # resampler.SetSize(pred_nii.GetSize())
    # resampler.SetOutputOrigin(pred_nii.GetOrigin())
    # resampler.SetOutputDirection(pred_nii.GetDirection())

    # resampled = resampler.Execute(pred_nii)
    resampled = resample_image(pred_nii, original_spacing, original_shape, is_label=True)  # Resample to original spacing
    resampled_pred = sitk.GetArrayFromImage(resampled)  # numpy: [D, H, W]

    # Convert to One-Hot
    num_classes = 2  # Assuming binary classification
    mask = torch.tensor(resampled_pred, dtype=torch.long)  # Convert to tensor
    one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=num_classes)  # Convert to one-hot encoding
    one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float()  # Permute to [C, D, H, W]

    # one_hot_mask = torch.nn.functional.one_hot(mask_tensor, num_classes=2).permute(3, 0, 1, 2)  # Convert to one-hot encoding and permute to [C, D, H, W]
    one_hot_mask = np.expand_dims(one_hot_mask, axis=0)  # Add batch dimension
    one_hot_mask = torch.tensor(one_hot_mask, dtype=torch.float32)  # Convert to tensor with float32 type
    return one_hot_mask  # Return the one-hot encoded mask tensor
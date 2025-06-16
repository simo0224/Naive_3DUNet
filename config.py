import argparse
import os

def get_config():
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_class', type=int, default=2, help='class number')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save logging')
    parser.add_argument('--model_ckpt_path', type=str, default="./model/ckpt.pth", help='model ckpt path')
    parser.add_argument('--if_load_model', action="store_true", help='whether to load model and optimizer')
    parser.add_argument('--if_save_nii', action="store_true", help='whether to save nii data')
    parser.add_argument('--if_use_tfm', action="store_true", help='whether to use tfm module')
    parser.add_argument('--comment', type=str, default="None", help='Comment for the training run')
    parser.add_argument('--DatasetDir', type=str, default="None", help='Comment for the training run')
    parser.add_argument('--mod', type=str, default="single", help='single or long')
    parser.add_argument('--save_nii_freq', type=int, default=20, help='save nii freq')
    parser.add_argument('--weight_loss_dice', type=float, default=1., help='weight for dice loss')
    parser.add_argument('--weight_loss_focal', type=float, default=1., help='weight for focal loss')
    parser.add_argument('--weight_loss_sim', type=float, default=0., help='weight for similarity loss')
    

    args = parser.parse_args()
    return args

# Example usage:
# from config_parser import get_config
# config = get_config()
# print(config.batch_size)
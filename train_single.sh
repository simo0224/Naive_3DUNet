# 训练参数配置（根据需要修改）
BATCH_SIZE=4
EPOCHS=160
LR=0.001
NUM_WORKERS=4
GPU_ID=7
OUTPUT_DIR="./output"
# DATA_DIR="/home/jincan/long_seg/my3DUNet/Data/Brats2024/Brats_Data/Train_split"
# DATA_DIR="/home/jincan/long_seg/my3DUNet/Data/Imaging/Imaging_for_3DUNet_single_new"
DATA_DIR="/home/jincan/long_seg/my3DUNet/Data/Imaging/Imaging_new_rig"
# 启动训练脚本
# CUDA_VISIBLE_DEVICES=1
python train.py \
    --mod "single" \
    --DatasetDir $DATA_DIR \
    --comment "single" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --gpu_id $GPU_ID \
    --output_dir $OUTPUT_DIR \
    --weight_loss_dice 1 \
    --weight_loss_focal 1 \
    --output_dir $OUTPUT_DIR \
    --weight_loss_sim 0 \
    --if_save_nii \
    --save_nii_freq 40 \
    # --if_load_model \
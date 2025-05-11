# 训练参数配置（根据需要修改）
BATCH_SIZE=2
EPOCHS=160
LR=0.001
NUM_WORKERS=4
GPU_ID=0
OUTPUT_DIR="./output"
DATA_DIR="/home/jincan/long_seg/my3DUNet/Data/Imaging/Imaging_new_rig"

# 启动训练脚本
# CUDA_VISIBLE_DEVICES=1
python train.py \
    --mod "long" \
    --DatasetDir $DATA_DIR \
    --comment "NONE" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --gpu_id $GPU_ID \
    --output_dir $OUTPUT_DIR \
    --weight_loss_dice 1.0 \
    --weight_loss_focal 1.0 \
    --weight_loss_sim 0.0 \
    --if_save_nii \
    --save_nii_freq 40 \
    # --if_load_model \
    # --if_use_tfm \

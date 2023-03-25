#!/bin/bash -l
#SBATCH --job-name=prepare_faces
#SBATCH -p psych_gpu
#SBATCH -G 1
#SBATCH --mem=20G
#SBATCH --array=0-9
#SBATCH -t 06:00:00

# train
# ./run.sh python data_preparation.py \
#     --img_folder /gpfs/milgram/pi/yildirim/shared_datasets/data_facecar/data_facecar/train \
#     --split_file /gpfs/milgram/pi/yildirim/shared_datasets/data_facecar/data_facecar/train_val_split.json \
#     --part_num $SLURM_ARRAY_TASK_COUNT \
#     --part_id $SLURM_ARRAY_TASK_ID

# val
./run.sh python data_preparation.py \
    --img_folder ~/scratch60/hakan/datasets/data_facecar/train \
    --split_file /gpfs/milgram/pi/yildirim/shared_datasets/data_facecar/data_facecar/train_val_split.json \
    --part_num $SLURM_ARRAY_TASK_COUNT \
    --part_id $SLURM_ARRAY_TASK_ID \
    --mode val

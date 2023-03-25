#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --mail-type ALL
#SBATCH --job-name train_faceEIG
#SBATCH -p psych_gpu
#SBATCH -n 1
#SBATCH -G 2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 4
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# export NCCL_SOCKET_IFNAME=^docker0,lo

# ./run.sh python train.py --name resnet50-last_fc --gpu_ids 0,1 --use_last_fc --net_recon resnet50
# ./run.sh python train.py --name resnet50 --gpu_ids 0,1 --net_recon resnet50
./run.sh python train.py --name vgg19-last_fc --gpu_ids 0,1 --use_last_fc --net_recon vgg19

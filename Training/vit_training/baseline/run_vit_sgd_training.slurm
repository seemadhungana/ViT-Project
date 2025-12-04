#!/bin/bash
#SBATCH --job-name=vit_sgd_100ep
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/data/p_dsi/dhungs1/vit_sgd_training_%j.out
#SBATCH --error=/data/p_dsi/dhungs1/vit_sgd_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seema.dhungana@vanderbilt.edu
echo "=========================================="
echo "ViT-Base ImageNet Training (SGD Optimizer)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo "Hyperparameters:"
echo "  Optimizer: SGD"
echo "  Learning rate: 0.1"
echo "  Momentum: 0.9"
echo "  Weight decay: 1e-4"
echo "  Batch size per GPU: 256"
echo "  Total epochs: 100"
echo "  Warmup epochs: 5"
echo "  Scheduler: CosineAnnealingLRWithWarmup"
echo "=========================================="
# Copy ImageNet to local /tmp for faster I/O
echo "Copying ImageNet to local storage..."
LOCAL_DATA_PATH=/tmp/imagenet_$SLURM_JOB_ID
mkdir -p $LOCAL_DATA_PATH
rsync -a --info=progress2 /data/p_dsi/dhungs1/imagenet/ $LOCAL_DATA_PATH/
echo "✓ Data copied to $LOCAL_DATA_PATH"
echo "=========================================="
# Container and bind paths
CONTAINER_PATH="/data/p_dsi/singularity-containers/pytorch_25.01-py3.sif"
# Distributed training setup
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
# Launch distributed training with proper bind mounts
singularity exec --nv --bind /data/p_dsi:/data/p_dsi,$LOCAL_DATA_PATH:$LOCAL_DATA_PATH \
    $CONTAINER_PATH torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /data/p_dsi/dhungs1/train_vit_sgd.py \
    --data_path $LOCAL_DATA_PATH \
    --output_dir /data/p_dsi/dhungs1/checkpoints_sgd_100ep \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --num_workers 16
echo "=========================================="
echo "Training finished at: $(date)"
echo "Cleaning up local storage..."
rm -rf $LOCAL_DATA_PATH
echo "✓ Cleanup complete"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=vit_measure_perturb
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/data/p_dsi/dhungs1/vit_measure_perturb_%j.out
#SBATCH --error=/data/p_dsi/dhungs1/vit_measure_perturb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seema.dhungana@vanderbilt.edu

echo "=========================================="
echo "ViT Single-Epoch Perturbation Effect Measurement"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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

# Configuration
BASELINE_CHECKPOINTS="/data/p_dsi/dhungs1/baseline_runs/checkpoints_sgd_100ep"
BASELINE_METRICS="/data/p_dsi/dhungs1/baseline_runs/checkpoints_sgd_100ep/training_metrics.csv"
OUTPUT_CSV="/data/p_dsi/dhungs1/perturbation_effects.csv"
THINGS_CSV="/data/p_dsi/dhungs1/things_inference.csv"
THINGS_IMG_DIR="/data/p_dsi/dhungs1/things_images"
THINGS_RDM_PATH="/data/p_dsi/dhungs1/RDM48_triplet.mat"
PERTURBATION_TYPES="gaussian uniform_gray label_shuffle target_noise"
PERTURB_EPOCHS="5 10 15 16 20 25 30 35 45 70 98"
EPSILON=0.1

echo "Configuration:"
echo "  Baseline checkpoints: $BASELINE_CHECKPOINTS"
echo "  Baseline metrics: $BASELINE_METRICS"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Perturbation types: $PERTURBATION_TYPES"
echo "  Epochs to test: $PERTURB_EPOCHS"
echo "  Epsilon (gaussian): $EPSILON"
echo "  Total measurements: 4 types × 11 epochs = 44 runs"
echo "=========================================="

# Launch distributed training
singularity exec --nv --bind /data/p_dsi:/data/p_dsi,$LOCAL_DATA_PATH:$LOCAL_DATA_PATH \
    $CONTAINER_PATH torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /data/p_dsi/dhungs1/measure_single_epoch_perturbation_effect.py \
    --baseline_checkpoint_dir $BASELINE_CHECKPOINTS \
    --baseline_metrics_csv $BASELINE_METRICS \
    --data_path $LOCAL_DATA_PATH \
    --output_csv $OUTPUT_CSV \
    --things_csv $THINGS_CSV \
    --things_img_dir $THINGS_IMG_DIR \
    --things_rdm_path $THINGS_RDM_PATH \
    --perturbation_types $PERTURBATION_TYPES \
    --perturb_epochs $PERTURB_EPOCHS \
    --epsilon $EPSILON \
    --batch_size 256 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --total_epochs 100 \
    --num_workers 16

echo "=========================================="
echo "Measurement finished at: $(date)"
echo "Cleaning up local storage..."
rm -rf $LOCAL_DATA_PATH
echo "✓ Cleanup complete"
echo "=========================================="

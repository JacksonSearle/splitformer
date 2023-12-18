#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --qos=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python3 ../train_model.py \
    --activation-dropout 0.2 \
    --checkpoint-activations False \
    --dropout 0.0 \
    --embed-dim 128 \
    --ffn-dim 1024 \
    --fsdp True \
    --layers 8 \
    --lr 0.001 \
    --model splitformer \
    --heads 8 \
    --seq-len 512 \
    --value-embed-dim 128 \
    --vocab-size 28783 \
    --device cuda \
    --epochs 30 \
    --batch-size 16 \
    --tokens-per-pass 10 \
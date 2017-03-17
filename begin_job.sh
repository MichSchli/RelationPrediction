#!/bin/bash


echo "Loading modules..."
module load gcc/4.9.2 
module load cuda/8.0.44 
module load cudnn/8.0-v5.1 python/3.5.0

export PYTHONPATH=:/nfs/home2/mschlic1/Projects/Converge/:/nfs/home2/mschlic1/Projects/Converge/Converge/

sbatch run-train_slurm.sh $1

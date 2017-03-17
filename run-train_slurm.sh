#!/bin/bash
#SBATCH --job-name=GCNcomplex
#SBATCH --time=24:00:00
#SBATCH --mail-user=michael.sejr@gmail.com
#SBATCH --mail-type=END
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH -N 2

DATASET="FB15k"
SETTINGS="gcn_basis.exp"

SCRIPT_DIR=~/Projects/RelationPrediction #$(dirname $(readlink -f $0))

VIRTUALENV_PATH=$SCRIPT_DIR"/venv"

TRAIN_PATH=$SCRIPT_DIR"/code/train.py"
DATASET_PATH=$SCRIPT_DIR"/data/"$DATASET
SETTINGS_PATH=$SCRIPT_DIR"/settings/"$SETTINGS

ARGUMENT_STRING="--settings "$SETTINGS_PATH" --dataset "$DATASET_PATH

#source $VIRTUALENV_PATH"/bin/activate"

python3 -u $TRAIN_PATH $ARGUMENT_STRING > $1'.out'

#deactivate

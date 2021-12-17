#!/usr/bin/env bash

DATASET="Toy"
SETTINGS=$1

SCRIPT_DIR=$(dirname $(readlink -f $0))

#VIRTUALENV_PATH=$SCRIPT_DIR"/venv"

TRAIN_PATH=$SCRIPT_DIR"/code/train.py"
DATASET_PATH=$SCRIPT_DIR"/data/"$DATASET
SETTINGS_PATH=$SCRIPT_DIR"/"$SETTINGS

ARGUMENT_STRING="--settings "$SETTINGS_PATH" --dataset "$DATASET_PATH

#source $VIRTUALENV_PATH"/bin/activate"

python $TRAIN_PATH $ARGUMENT_STRING

#deactivate

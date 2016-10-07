#!bin/bash

ALGORITHM=$1

TRAIN_FILE="data/Toy/toy-train.txt"
VALIDATION_FILE="data/Toy/toy-test.txt"
ENTITY_DICTIONARY="data/Toy/entities.dic"
RELATION_DICTIONARY="data/Toy/relations.dic"
MODEL_PATH='models/toy-'$ALGORITHM'.model'

THEANO_FLAGS='floatX=float32,warn_float64=raise,optimizer_including=local_remove_all_assert' python code/experts/train.py --train_data $TRAIN_FILE --validation_data $VALIDATION_FILE --entities $ENTITY_DICTIONARY --relations $RELATION_DICTIONARY --model_path $MODEL_PATH --algorithm $ALGORITHM

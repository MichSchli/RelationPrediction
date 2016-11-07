#!bin/bash

ALGORITHM=$1
DATASET=$2

TRAIN_FILE='data/'$DATASET'/train.txt'
VALIDATION_FILE='data'/$DATASET'/valid.txt'

#No access to test allowed at training time:
TEST_FILE='data/'$DATASET'/valid.txt'

ENTITY_DICTIONARY='data/'$DATASET'/entities.dict'
RELATION_DICTIONARY='data/'$DATASET/'relations.dict'
MODEL_PATH="models/$1.model"

THEANO_FLAGS='floatX=float32,exception_verbosity=high,warn_float64=raise,optimizer_including=local_remove_all_assert' python code/experts/train-expert.py --train_data $TRAIN_FILE --validation_data $VALIDATION_FILE --test_data $TEST_FILE --entities $ENTITY_DICTIONARY --relations $RELATION_DICTIONARY --model_path $MODEL_PATH --algorithm $ALGORITHM

#!bin/bash

TRAIN_FILE="data/FB15k/freebase_mtr100_mte100-train.txt"
VALIDATION_FILE="data/FB15k/freebase_mtr100_mte100-valid.txt"
TEST_FILE="data/FB15k/freebase_mtr100_mte100-test.txt"
ENTITY_DICTIONARY="data/FB15k/entities.dict"
RELATION_DICTIONARY="data/FB15k/relations.dict"
MODEL_PATH="models/distmult.model"
ALGORITHM="distmult"
PREDICTION_FILE="data/temporary.txt"

THEANO_FLAGS='floatX=float32,warn_float64=raise,optimizer_including=local_remove_all_assert' python code/parsing/predict.py --train_data $TRAIN_FILE --validation_data $VALIDATION_FILE --test_data $TEST_FILE --entities $ENTITY_DICTIONARY --relations $RELATION_DICTIONARY --model_path $MODEL_PATH --algorithm $ALGORITHM --prediction_file $PREDICTION_FILE

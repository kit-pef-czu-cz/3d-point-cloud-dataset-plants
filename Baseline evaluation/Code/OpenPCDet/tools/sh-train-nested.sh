#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG=$1
OUTPUT=$2
EPOCHS=$3
DATA_NAME=$4
ES_PAT=$5
ES_WARM=$6
ES_METRIC=$7
ES_OBJ=$8

DATA="../data/"
OUTERS=(0 1 2 3 4)
for o in ${OUTERS[@]}; do
	INNERS=${OUTERS[@]/$o}
	DATASETS=()
	for i in ${INNERS[@]}; do
		DATASETS+=( "$DATA_NAME-t$o-v$i" )
	done
	python run_train.py $CONFIG $OUTPUT/${DATASETS[0]} -e $EPOCHS -d $DATA${DATASETS[0]} -g 0 --es-patience $ES_PAT --es-warmup $ES_WARM --es-objective $ES_METRIC $ES_OBJ &
	python run_train.py $CONFIG $OUTPUT/${DATASETS[1]} -e $EPOCHS -d $DATA${DATASETS[1]} -g 1 --es-patience $ES_PAT --es-warmup $ES_WARM --es-objective $ES_METRIC $ES_OBJ &
	python run_train.py $CONFIG $OUTPUT/${DATASETS[2]} -e $EPOCHS -d $DATA${DATASETS[2]} -g 2 --es-patience $ES_PAT --es-warmup $ES_WARM --es-objective $ES_METRIC $ES_OBJ &
	python run_train.py $CONFIG $OUTPUT/${DATASETS[3]} -e $EPOCHS -d $DATA${DATASETS[3]} -g 3 --es-patience $ES_PAT --es-warmup $ES_WARM --es-objective $ES_METRIC $ES_OBJ &
	wait
done

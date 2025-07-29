#!/bin/bash
CONFIG=$1
METHOD=$2

OUTERS=(0 1 2 3 4)
for o in ${OUTERS[@]}; do
	INNERS=${OUTERS[@]/$o}
	for i in ${INNERS[@]}; do
		python pcdet/datasets/custom/custom_dataset.py $1 -d "$2-t$o-v$i" &
	done
	wait
done

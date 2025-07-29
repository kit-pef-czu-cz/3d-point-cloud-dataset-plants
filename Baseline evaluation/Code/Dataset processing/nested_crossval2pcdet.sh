#!/bin/bash
DATASET=$1
CSV_PATH=$2
METHOD=$3
OUTPUT=$4

OUTERS=(0 1 2 3 4)
for o in ${OUTERS[@]}; do
        INNERS=${OUTERS[@]/$o}
        for i in ${INNERS[@]}; do
                python crossval2pcdet.py $1 "$2/$3-t$o-v$i.csv" "$4/$3-t$o-v$i" &
        done
        wait
done

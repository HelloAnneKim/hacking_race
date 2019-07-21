#!/bin/bash
WORK_DIR=~/viz-data
mkdir -p $WORK_DIR
for SUFFIX in recoded_1000G.noadmixed.mat recoded_1000G.raw.noadmixed.lbls3_3 recoded_1000G.raw.noadmixed.lbls3 recoded_1000G.raw.noadmixed.ids
do
wget http://lovingscience.com/ancestries/downloads/$SUFFIX -O $WORK_DIR/$SUFFIX
done


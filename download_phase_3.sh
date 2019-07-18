#! /bin/bash
WORK_DIR=~/genomes
mkdir $WORK_DIR
lftp ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/  -e "mirror --continue ./ $WORK_DIR"

#!/bin/bash
for rep in 1 2 3 4 5 ; do
ntasks=( 32 16 8 4 2 )
for i in "${ntasks[@]}"
do
sed "s/§NTASKS§/$i/" batch_template.sh | sed "s/§REP§/$rep/" > batch.sh.tmp
sbatch batch.sh.tmp
sleep 3
done
done
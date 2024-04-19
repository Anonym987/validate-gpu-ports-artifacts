#!/bin/bash
for rep in 1 2 3 4 5 ; do
ntasks=( 2 4 8 16 32  )
for i in "${ntasks[@]}"
do
sed "s/§NTASKS§/$i/" batch_template.sh | sed "s/§REP§/$rep/" > batch.sh.tmp
sbatch batch.sh.tmp
done
done
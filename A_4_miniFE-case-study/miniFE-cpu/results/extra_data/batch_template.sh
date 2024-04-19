#!/bin/bash
#SBATCH -J mFE§NTASKS§_§REP§
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /p/project/deepsea/geiss1/miniFE-2.2.0/openmp-opt/results/logs/mFE-cpu.§NTASKS§_§REP§_%A_%a.err
#SBATCH -o /p/project/deepsea/geiss1/miniFE-2.2.0/openmp-opt/results/logs/mFE-cpu.§NTASKS§_§REP§_%A_%a.out
#
#SBATCH -a 1-5 #1-5
#SBATCH -A deepsea
#SBATCH -p dp-cn
#SBATCH --ntasks-per-node=1
#SBATCH -n §NTASKS§       # 1 Prozess
#SBATCH -c 48       # 16 Kerne pro Prozess
#SBATCH --mem=46000    # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 01:30:00    # in Stunden und Minuten, oder '#SBATCH -t 10' - nur Minuten

ml --force purge
ml use $OTHERSTAGES
ml Stages/2023

ml GCC ParaStationMPI Score-P

export OMP_NUM_THREADS=24
export OMP_PLACES="cores(24)"
export OMP_AFFINITY_FORMAT="%H %N: %n %A"
export OMP_DISPLAY_AFFINITY=FALSE
export TMPDIR=/scratch

xSizes=( [2]=2 [4]=2 [8]=4 [16]=4 [32]=8 )
ySizes=( [2]=1 [4]=2 [8]=2 [16]=4 [32]=4 )
zSizes=( [2]=1 [4]=1 [8]=1 [16]=1 [32]=1 )

x=${xSizes[SLURM_NTASKS]}
y=${ySizes[SLURM_NTASKS]}
z=${zSizes[SLURM_NTASKS]}

m=$((150+50*SLURM_ARRAY_TASK_ID))

n=$((m*x*m*y*m*z))
export SCOREP_EXPERIMENT_DIRECTORY="mFE-cpu.p${SLURM_NTASKS}.n$((m*m*m)).r§REP§" 
export EXTRA_PROF_EXPERIMENT_DIRECTORY=$SCOREP_EXPERIMENT_DIRECTORY
srun -c 48 ../../sources/src-ai/miniFE.x -nx $((m*x)) -ny $((m*y)) -nz $((m*z))
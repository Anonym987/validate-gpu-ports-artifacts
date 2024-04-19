#!/bin/bash

#export SCOREP_ENABLE_UNWINDING=true
export SCOREP_METRIC_PAPI="PAPI_DP_OPS"
export SCOREP_METRIC_PAPI_PER_PROCESS="skx_unc_imc0::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc1::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc2::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc3::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc4::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc5::UNC_M_CAS_COUNT:ALL:cpu=0,skx_unc_imc0::UNC_M_CAS_COUNT:ALL:cpu=12,skx_unc_imc1::UNC_M_CAS_COUNT:ALL:cpu=12,skx_unc_imc2::UNC_M_CAS_COUNT:ALL:cpu=12,skx_unc_imc3::UNC_M_CAS_COUNT:ALL:cpu=12,skx_unc_imc4::UNC_M_CAS_COUNT:ALL:cpu=12,skx_unc_imc5::UNC_M_CAS_COUNT:ALL:cpu=12"
export SCOREP_FILTERING_FILE="visits0.001.scorep-filter"

rep="AI"
ntasks=( 2 4 8 16 32 )
for i in "${ntasks[@]}"
do
sed "s/§NTASKS§/$i/" batch_template.sh | sed "s/§REP§/$rep/" > batch.sh.tmp
sbatch batch.sh.tmp
done

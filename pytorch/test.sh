# source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
# export LD_LIBRARY_PATH=/opt/intel/oneapi/intelpython/python3.7/envs/pytorch/lib/python3.7/site-packages/torch/lib/

CONDA_PREFIX=/opt/intel/oneapi/intelpython/latest/envs/rnnt
# export OMP_NUM_THREADS=16
# export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so"
# source /opt/intel/oneapi/pytorch/1.8.0/lib/python3.7/site-packages/env/setvars.sh
# source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
# Example:
# Run 2 processes on 2 sockets. (28 cores/socket, 4 cores for CCL, 24 cores for computation)
#
# CCL_WORKER_COUNT means per instance threads used by CCL.
# CCL_WORKER_COUNT, CCL_WORKER_AFFINITY and I_MPI_PIN_DOMAIN should be consistent.
export CCL_WORKER_COUNT=2
export CCL_WORKER_AFFINITY="0,1,18,19"
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="0,1,18,19"

# mpiexec.hydra -np 2 -ppn 2 -l -hosts sr112 -genv I_MPI_PIN_DOMAIN=[0x3fffc,0xffff00000,] \
#               -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0      \
#               -genv OMP_NUM_THREADS=16 ${CONDA_PREFIX}/bin/python -u test_ddp.py



mpiexec.hydra -np 2 -ppn 2 -hosts sr112 -genv I_MPI_PIN_DOMAIN [0x3fffc,0xffff00000,] \
  -genv KMP_BLOCKTIME 1 -genv KMP_AFFINITY granularity=fine,compact,1,0 -genv OMP_NUM_THREADS 16 \
  -map-by socket -print-rank-map \
  ${CONDA_PREFIX}/bin/python -u test_ddp.py
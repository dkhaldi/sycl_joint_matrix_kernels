# Unified GEMM on Nvidia Tensor Cores, Intel XMX of PVC and DG2, and Intel AMX of SPR  using SYCL joint matrix

## joint_matrix_fill_k_cache.cpp:
#### Portable Optimizations:
 - cache tiling of i and j
 - cache tiling on k as well (so no reordering is needed)
 - data reuse of A and B in physical layer
#### Specific Optimizations for PVC:
 - Out of Bounds checking is used for PVC using -DOOB
 - Prefetch for PVC is enabled under -DPREFETCH
#### Specific options for AMX and SG2
 - Both row major and VNNI transform options. For row major ommit -DVNNI
#### Missing optimizations:
no reordering, no SLM for DG2/Nvidia
#### Important:
For maximum performance, cache and registers blocking parameters are
different between Nvidia Tensor Cores, AMX and DPAS of DG2 vs PVC. See
specific parameters below:

M=N=K=X cases, use -DMATRIX_SIZE=X
Otherwise, use: -DMATRIX_M=1024 -DMATRIX_N=6144 -DMATRIX_K=6144

## Build Command lines

### Nvidia (~70 Tflops) Add  -DNVIDIA
#### 2048
icpx -fsycl -fsycl-targets=nvidia_gpu_sm_80 joint_matrix_fill_k_cache.cpp  -DNVIDIA -DMCACHE1=64 -DNCACHE1=64 -DMCACHE2=128 -DNCACHE2=128

#### 4096
icpx -fsycl -fsycl-targets=nvidia_gpu_sm_80 joint_matrix_fill_k_cache.cpp -DMATRIX_SIZE=4096  -DNVIDIA -DMCACHE1=64 -DNCACHE1=64 -DMCACHE2=128 -DNCACHE2=128

### PVC row major (~220 TFlops)
#### 2048
icpx -fsycl joint_matrix_fill_k_cache.cpp -DPREFETCH -DOOB

#### 4096 VNNI
icpx -fsycl joint_matrix_fill_k_cache.cpp -DPREFETCH -DOOB -DMATRIX_SIZE=4096

### DG2 VNNI (~45 Tflops)
#### 2048
icpx -fsycl joint_matrix_fill_k_cache.cpp -DNCACHE1=32 -DMCACHE2=128 -DNCACHE2=128 -DKCACHE2=16 -DVNNI
#### 4096 VNNI
icpx -fsycl joint_matrix_fill_k_cache.cpp -DNCACHE1=32 -DMCACHE2=128 -DNCACHE2=128 -DKCACHE2=16 -DMATRIX_SIZE=4096 -DVNNI

### SPR VNNI (~60 Tflops)
#### 2048
icpx -fsycl joint_matrix_fill_k_cache.cpp -DNCACHE1=32 -DKCACHE1=32 -DMCACHE2=128 -DNCACHE2=128 -DKCACHE2=1024 -DVNNI
#### 4096
icpx -fsycl joint_matrix_fill_k_cache.cpp -DNCACHE1=32 -DKCACHE1=32 -DMCACHE2=256 -DNCACHE2=256 -DKCACHE2=1024 -DMATRIX_SIZE=4096 -DVNNI

## Execution command lines
### To run on Nvidia GPU:
ONEAPI_DEVICE_SELECTOR=cuda:0  ./a.out

### To run on Intel GPU:
SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file" ./a.out
###
To run on CPU:
DPCPP_CPU_NUM_CUS=112 ./a.out

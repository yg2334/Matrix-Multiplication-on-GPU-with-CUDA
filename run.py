prog = """
// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;
#define minx(a,b) a < b ? a: b
#define AS(i,j) AS[i*TILEDIM_K + j]
#define BS(i,j) BS[i*TILEDIM_N + j]

#include <stdio.h>


__device__ static inline _FTYPE_ getVal(int i, int j, int N, const _FTYPE_ * mat) {
    if(i < N && j < N) {
        return mat[i*N + j];
    }
    return 0.0;
}
__device__ static inline void setVal(int i, int j, int N, _FTYPE_ val, _FTYPE_* mat) {
    if(i < N && j < N) {
        mat[i*N + j] = val;
    }
}

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
     extern __shared__ _FTYPE_ smem[];

    _FTYPE_* AS = (_FTYPE_*) smem; // TILEDIM_M * TILEDIM_K
    _FTYPE_* BS = (_FTYPE_*) (smem + TILEDIM_M*TILEDIM_K); // TILEDIM_k * TILEDIM_M

    int I =  blockIdx.y*TILEDIM_M + threadIdx.y;
    int J =  blockIdx.x*TILEDIM_N + threadIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bdimx = blockDim.x;
    int bdimy = blockDim.y;
    _FTYPE_ _c[%s] = %s;
    
    #pragma unroll
    for(unsigned int kk = 0; kk < ((N+TILEDIM_K -  1)/TILEDIM_K); kk++ ) {
        %s
    
        %s
 
        __syncthreads();
        
        unsigned int iterSize = minx((N - kk*TILEDIM_K),TILEDIM_K);
        if ( I < N && J < N) {
            #pragma unroll
            for(int k = 0; k < iterSize;k++) {
               %s
            }
        }
        __syncthreads();
    }
    %s
}
#endif
"""
PER_THREAD_WIDTH = 2
TILESCALE_M = 0
TILESCALE_N = 0
TILESCALE_K = 0
TILESTEP_M = 0
TILESTEP_N = 0
TILESTEP_AK = 0
TILESTEP_BK = 0
with open('./src_todo_T4/mytypes.h','r') as f:
    for line in f.readlines():
        if "PER_THREAD_WIDTH" in line:
            PER_THREAD_WIDTH = int(line.strip().split()[-1])
        elif "TILESCALE_M" in line and "#define" in line:
            TILESCALE_M = int(line.strip().split()[-1])
        elif "TILESCALE_N" in line and "#define" in line:
            TILESCALE_N = int(line.strip().split()[-1])
        elif "TILESCALE_K" in line and "#define" in line:
            TILESCALE_K = int(line.strip().split()[-1])
        elif "TILESTEP_M" in line and "#define" in line:
            TILESTEP_M = int(line.strip().split()[-1])
        elif "TILESTEP_N" in line and "#define" in line:
            TILESTEP_N = int(line.strip().split()[-1])
        elif "TILESTEP_AK" in line and "#define" in line:
            TILESTEP_AK = int(line.strip().split()[-1])
        elif "TILESTEP_BK" in line and "#define" in line:
            TILESTEP_BK = int(line.strip().split()[-1])
        
    

def get_c_init(TILESTEP_M,TILESTEP_N):
    a = ["0"]*TILESCALE_M*TILESCALE_N
    res = ",".join(a)
    return f"{{{res}}}"

def get_A(TILESTEP_M,TILESTEP_K):
    res = ""
    for i in range(TILESTEP_M):
        for j in range(TILESTEP_K):
            res += f"AS((ty + {i}*bdimy),(tx + {j}*bdimx)) = getVal(I + {i}*bdimy,kk*TILEDIM_K + tx + {j}*bdimx,N,A);\n"
    return res

def get_B(TILESTEP_N, TILESTEP_K):
    res = ""
    for i in range(TILESTEP_K):
        for j in range(TILESTEP_N):
            res += f"BS((ty + {i}*bdimy),(tx + {j}*bdimx)) = getVal(kk*TILEDIM_K + ty + {i}*bdimy,J + {j}*bdimx,N,B);\n"
    return res

def get_C(TILESCALE_M,TILESCALE_N):
    res = ""
    for i in range(TILESCALE_M):
        for j in range(TILESCALE_N):
            res += f"_c[{i*TILESCALE_N + j}] += AS((ty + {i}*bdimy),(k)) *BS((k),(tx + {j}*bdimx));\n"
    return res

def get_set_Val(TILESCALE_M,TILESCALE_N):
    res = ""
    for i in range(TILESCALE_M):
        for j in range(TILESCALE_N):
            res += f"setVal(I + {i}*bdimy, J + {j}*bdimx ,N, _c[{i*TILESCALE_N + j}], C);\n"
    return res


result = prog % (TILESTEP_M*TILESCALE_N,
                 get_c_init(TILESTEP_M,TILESTEP_N), 
                 get_A(TILESTEP_M,TILESTEP_AK), 
                 get_B(TILESTEP_N,TILESTEP_BK), 
                 get_C(TILESCALE_M,TILESCALE_N), 
                 get_set_Val(TILESCALE_M,TILESCALE_N))
print(result)
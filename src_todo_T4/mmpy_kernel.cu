
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
    _FTYPE_ _c[64] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    #pragma unroll
    for(unsigned int kk = 0; kk < ((N+TILEDIM_K -  1)/TILEDIM_K); kk++ ) {
        AS((ty + 0*bdimy),(tx + 0*bdimx)) = getVal(I + 0*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 0*bdimy),(tx + 1*bdimx)) = getVal(I + 0*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 1*bdimy),(tx + 0*bdimx)) = getVal(I + 1*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 1*bdimy),(tx + 1*bdimx)) = getVal(I + 1*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 2*bdimy),(tx + 0*bdimx)) = getVal(I + 2*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 2*bdimy),(tx + 1*bdimx)) = getVal(I + 2*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 3*bdimy),(tx + 0*bdimx)) = getVal(I + 3*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 3*bdimy),(tx + 1*bdimx)) = getVal(I + 3*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 4*bdimy),(tx + 0*bdimx)) = getVal(I + 4*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 4*bdimy),(tx + 1*bdimx)) = getVal(I + 4*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 5*bdimy),(tx + 0*bdimx)) = getVal(I + 5*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 5*bdimy),(tx + 1*bdimx)) = getVal(I + 5*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 6*bdimy),(tx + 0*bdimx)) = getVal(I + 6*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 6*bdimy),(tx + 1*bdimx)) = getVal(I + 6*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 7*bdimy),(tx + 0*bdimx)) = getVal(I + 7*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 7*bdimy),(tx + 1*bdimx)) = getVal(I + 7*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 8*bdimy),(tx + 0*bdimx)) = getVal(I + 8*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 8*bdimy),(tx + 1*bdimx)) = getVal(I + 8*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 9*bdimy),(tx + 0*bdimx)) = getVal(I + 9*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 9*bdimy),(tx + 1*bdimx)) = getVal(I + 9*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 10*bdimy),(tx + 0*bdimx)) = getVal(I + 10*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 10*bdimy),(tx + 1*bdimx)) = getVal(I + 10*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 11*bdimy),(tx + 0*bdimx)) = getVal(I + 11*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 11*bdimy),(tx + 1*bdimx)) = getVal(I + 11*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 12*bdimy),(tx + 0*bdimx)) = getVal(I + 12*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 12*bdimy),(tx + 1*bdimx)) = getVal(I + 12*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 13*bdimy),(tx + 0*bdimx)) = getVal(I + 13*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 13*bdimy),(tx + 1*bdimx)) = getVal(I + 13*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 14*bdimy),(tx + 0*bdimx)) = getVal(I + 14*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 14*bdimy),(tx + 1*bdimx)) = getVal(I + 14*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);
AS((ty + 15*bdimy),(tx + 0*bdimx)) = getVal(I + 15*bdimy,kk*TILEDIM_K + tx + 0*bdimx,N,A);
AS((ty + 15*bdimy),(tx + 1*bdimx)) = getVal(I + 15*bdimy,kk*TILEDIM_K + tx + 1*bdimx,N,A);

    
        BS((ty + 0*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 0*bdimy,J + 0*bdimx,N,B);
BS((ty + 0*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 0*bdimy,J + 1*bdimx,N,B);
BS((ty + 0*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 0*bdimy,J + 2*bdimx,N,B);
BS((ty + 0*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 0*bdimy,J + 3*bdimx,N,B);
BS((ty + 1*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 1*bdimy,J + 0*bdimx,N,B);
BS((ty + 1*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 1*bdimy,J + 1*bdimx,N,B);
BS((ty + 1*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 1*bdimy,J + 2*bdimx,N,B);
BS((ty + 1*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 1*bdimy,J + 3*bdimx,N,B);
BS((ty + 2*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 2*bdimy,J + 0*bdimx,N,B);
BS((ty + 2*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 2*bdimy,J + 1*bdimx,N,B);
BS((ty + 2*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 2*bdimy,J + 2*bdimx,N,B);
BS((ty + 2*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 2*bdimy,J + 3*bdimx,N,B);
BS((ty + 3*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 3*bdimy,J + 0*bdimx,N,B);
BS((ty + 3*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 3*bdimy,J + 1*bdimx,N,B);
BS((ty + 3*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 3*bdimy,J + 2*bdimx,N,B);
BS((ty + 3*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 3*bdimy,J + 3*bdimx,N,B);
BS((ty + 4*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 4*bdimy,J + 0*bdimx,N,B);
BS((ty + 4*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 4*bdimy,J + 1*bdimx,N,B);
BS((ty + 4*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 4*bdimy,J + 2*bdimx,N,B);
BS((ty + 4*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 4*bdimy,J + 3*bdimx,N,B);
BS((ty + 5*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 5*bdimy,J + 0*bdimx,N,B);
BS((ty + 5*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 5*bdimy,J + 1*bdimx,N,B);
BS((ty + 5*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 5*bdimy,J + 2*bdimx,N,B);
BS((ty + 5*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 5*bdimy,J + 3*bdimx,N,B);
BS((ty + 6*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 6*bdimy,J + 0*bdimx,N,B);
BS((ty + 6*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 6*bdimy,J + 1*bdimx,N,B);
BS((ty + 6*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 6*bdimy,J + 2*bdimx,N,B);
BS((ty + 6*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 6*bdimy,J + 3*bdimx,N,B);
BS((ty + 7*bdimy),(tx + 0*bdimx)) = getVal(kk*TILEDIM_K + ty + 7*bdimy,J + 0*bdimx,N,B);
BS((ty + 7*bdimy),(tx + 1*bdimx)) = getVal(kk*TILEDIM_K + ty + 7*bdimy,J + 1*bdimx,N,B);
BS((ty + 7*bdimy),(tx + 2*bdimx)) = getVal(kk*TILEDIM_K + ty + 7*bdimy,J + 2*bdimx,N,B);
BS((ty + 7*bdimy),(tx + 3*bdimx)) = getVal(kk*TILEDIM_K + ty + 7*bdimy,J + 3*bdimx,N,B);

 
        __syncthreads();
        
        unsigned int iterSize = minx((N - kk*TILEDIM_K),TILEDIM_K);
        if ( I < N && J < N) {
            #pragma unroll
            for(int k = 0; k < iterSize;k++) {
               _c[0] += AS((ty + 0*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[1] += AS((ty + 0*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[2] += AS((ty + 0*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[3] += AS((ty + 0*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[4] += AS((ty + 1*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[5] += AS((ty + 1*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[6] += AS((ty + 1*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[7] += AS((ty + 1*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[8] += AS((ty + 2*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[9] += AS((ty + 2*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[10] += AS((ty + 2*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[11] += AS((ty + 2*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[12] += AS((ty + 3*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[13] += AS((ty + 3*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[14] += AS((ty + 3*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[15] += AS((ty + 3*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[16] += AS((ty + 4*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[17] += AS((ty + 4*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[18] += AS((ty + 4*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[19] += AS((ty + 4*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[20] += AS((ty + 5*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[21] += AS((ty + 5*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[22] += AS((ty + 5*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[23] += AS((ty + 5*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[24] += AS((ty + 6*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[25] += AS((ty + 6*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[26] += AS((ty + 6*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[27] += AS((ty + 6*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[28] += AS((ty + 7*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[29] += AS((ty + 7*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[30] += AS((ty + 7*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[31] += AS((ty + 7*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[32] += AS((ty + 8*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[33] += AS((ty + 8*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[34] += AS((ty + 8*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[35] += AS((ty + 8*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[36] += AS((ty + 9*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[37] += AS((ty + 9*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[38] += AS((ty + 9*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[39] += AS((ty + 9*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[40] += AS((ty + 10*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[41] += AS((ty + 10*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[42] += AS((ty + 10*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[43] += AS((ty + 10*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[44] += AS((ty + 11*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[45] += AS((ty + 11*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[46] += AS((ty + 11*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[47] += AS((ty + 11*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[48] += AS((ty + 12*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[49] += AS((ty + 12*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[50] += AS((ty + 12*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[51] += AS((ty + 12*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[52] += AS((ty + 13*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[53] += AS((ty + 13*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[54] += AS((ty + 13*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[55] += AS((ty + 13*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[56] += AS((ty + 14*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[57] += AS((ty + 14*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[58] += AS((ty + 14*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[59] += AS((ty + 14*bdimy),(k)) *BS((k),(tx + 3*bdimx));
_c[60] += AS((ty + 15*bdimy),(k)) *BS((k),(tx + 0*bdimx));
_c[61] += AS((ty + 15*bdimy),(k)) *BS((k),(tx + 1*bdimx));
_c[62] += AS((ty + 15*bdimy),(k)) *BS((k),(tx + 2*bdimx));
_c[63] += AS((ty + 15*bdimy),(k)) *BS((k),(tx + 3*bdimx));

            }
        }
        __syncthreads();
    }
    setVal(I + 0*bdimy, J + 0*bdimx ,N, _c[0], C);
setVal(I + 0*bdimy, J + 1*bdimx ,N, _c[1], C);
setVal(I + 0*bdimy, J + 2*bdimx ,N, _c[2], C);
setVal(I + 0*bdimy, J + 3*bdimx ,N, _c[3], C);
setVal(I + 1*bdimy, J + 0*bdimx ,N, _c[4], C);
setVal(I + 1*bdimy, J + 1*bdimx ,N, _c[5], C);
setVal(I + 1*bdimy, J + 2*bdimx ,N, _c[6], C);
setVal(I + 1*bdimy, J + 3*bdimx ,N, _c[7], C);
setVal(I + 2*bdimy, J + 0*bdimx ,N, _c[8], C);
setVal(I + 2*bdimy, J + 1*bdimx ,N, _c[9], C);
setVal(I + 2*bdimy, J + 2*bdimx ,N, _c[10], C);
setVal(I + 2*bdimy, J + 3*bdimx ,N, _c[11], C);
setVal(I + 3*bdimy, J + 0*bdimx ,N, _c[12], C);
setVal(I + 3*bdimy, J + 1*bdimx ,N, _c[13], C);
setVal(I + 3*bdimy, J + 2*bdimx ,N, _c[14], C);
setVal(I + 3*bdimy, J + 3*bdimx ,N, _c[15], C);
setVal(I + 4*bdimy, J + 0*bdimx ,N, _c[16], C);
setVal(I + 4*bdimy, J + 1*bdimx ,N, _c[17], C);
setVal(I + 4*bdimy, J + 2*bdimx ,N, _c[18], C);
setVal(I + 4*bdimy, J + 3*bdimx ,N, _c[19], C);
setVal(I + 5*bdimy, J + 0*bdimx ,N, _c[20], C);
setVal(I + 5*bdimy, J + 1*bdimx ,N, _c[21], C);
setVal(I + 5*bdimy, J + 2*bdimx ,N, _c[22], C);
setVal(I + 5*bdimy, J + 3*bdimx ,N, _c[23], C);
setVal(I + 6*bdimy, J + 0*bdimx ,N, _c[24], C);
setVal(I + 6*bdimy, J + 1*bdimx ,N, _c[25], C);
setVal(I + 6*bdimy, J + 2*bdimx ,N, _c[26], C);
setVal(I + 6*bdimy, J + 3*bdimx ,N, _c[27], C);
setVal(I + 7*bdimy, J + 0*bdimx ,N, _c[28], C);
setVal(I + 7*bdimy, J + 1*bdimx ,N, _c[29], C);
setVal(I + 7*bdimy, J + 2*bdimx ,N, _c[30], C);
setVal(I + 7*bdimy, J + 3*bdimx ,N, _c[31], C);
setVal(I + 8*bdimy, J + 0*bdimx ,N, _c[32], C);
setVal(I + 8*bdimy, J + 1*bdimx ,N, _c[33], C);
setVal(I + 8*bdimy, J + 2*bdimx ,N, _c[34], C);
setVal(I + 8*bdimy, J + 3*bdimx ,N, _c[35], C);
setVal(I + 9*bdimy, J + 0*bdimx ,N, _c[36], C);
setVal(I + 9*bdimy, J + 1*bdimx ,N, _c[37], C);
setVal(I + 9*bdimy, J + 2*bdimx ,N, _c[38], C);
setVal(I + 9*bdimy, J + 3*bdimx ,N, _c[39], C);
setVal(I + 10*bdimy, J + 0*bdimx ,N, _c[40], C);
setVal(I + 10*bdimy, J + 1*bdimx ,N, _c[41], C);
setVal(I + 10*bdimy, J + 2*bdimx ,N, _c[42], C);
setVal(I + 10*bdimy, J + 3*bdimx ,N, _c[43], C);
setVal(I + 11*bdimy, J + 0*bdimx ,N, _c[44], C);
setVal(I + 11*bdimy, J + 1*bdimx ,N, _c[45], C);
setVal(I + 11*bdimy, J + 2*bdimx ,N, _c[46], C);
setVal(I + 11*bdimy, J + 3*bdimx ,N, _c[47], C);
setVal(I + 12*bdimy, J + 0*bdimx ,N, _c[48], C);
setVal(I + 12*bdimy, J + 1*bdimx ,N, _c[49], C);
setVal(I + 12*bdimy, J + 2*bdimx ,N, _c[50], C);
setVal(I + 12*bdimy, J + 3*bdimx ,N, _c[51], C);
setVal(I + 13*bdimy, J + 0*bdimx ,N, _c[52], C);
setVal(I + 13*bdimy, J + 1*bdimx ,N, _c[53], C);
setVal(I + 13*bdimy, J + 2*bdimx ,N, _c[54], C);
setVal(I + 13*bdimy, J + 3*bdimx ,N, _c[55], C);
setVal(I + 14*bdimy, J + 0*bdimx ,N, _c[56], C);
setVal(I + 14*bdimy, J + 1*bdimx ,N, _c[57], C);
setVal(I + 14*bdimy, J + 2*bdimx ,N, _c[58], C);
setVal(I + 14*bdimy, J + 3*bdimx ,N, _c[59], C);
setVal(I + 15*bdimy, J + 0*bdimx ,N, _c[60], C);
setVal(I + 15*bdimy, J + 1*bdimx ,N, _c[61], C);
setVal(I + 15*bdimy, J + 2*bdimx ,N, _c[62], C);
setVal(I + 15*bdimy, J + 3*bdimx ,N, _c[63], C);

}
#endif


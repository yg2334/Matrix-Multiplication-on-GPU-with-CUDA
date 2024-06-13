#ifndef MYTYPES_H
#define MYTYPES_H

// tilescale (# of points computed by each thread)
#ifndef TILESCALE_M
#define TILESCALE_M 16
#endif
#ifndef TILESCALE_N
#define TILESCALE_N 4
#endif
#ifndef TILESCALE_K
#define TILESCALE_K 4
#endif

#define TILEDIM_M 128
#define TILEDIM_N 128

// matrix A loads
// with warps along the horiziontal axis (K)
// so to get good coalescaed loads, we want TILEDIM_K to be >= 32
//
#define TILEDIM_K 64 // Enter your own values

// step size in each dimension
// TN = DIM_N/bx, TM = DIM_M/by, AK = TK/BX, BK = TK/BY 
#define TILESTEP_N 4
#define TILESTEP_AK 2
#define TILESTEP_BK 8
#define TILESTEP_M 16
#define PER_THREAD_WIDTH 2
#ifndef BLOCKDIM_X
// should be matching bx and by
#define SHARED_BLOCK 16
#else 
#define SHARED_BLOCK BLOCKDIM_X
#endif
#endif

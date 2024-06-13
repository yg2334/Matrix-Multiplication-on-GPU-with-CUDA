
#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   const int bx = TILEDIM_M;
   const int by = TILEDIM_N;
   // set your block dimensions and grid dimensions here
   gridDim.x = n / bx;
   gridDim.y = n / by;

   // you can overwrite blockDim here if you like.
   if (n % bx != 0)
      gridDim.x++;
   if (n % by != 0)
      gridDim.y++;
}

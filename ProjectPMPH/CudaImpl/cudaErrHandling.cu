#ifndef CUDA_ERR_HANDLING
#define CUDA_ERR_HANDLING

//#include <cuda.h>

// Inspired by
// http://stackoverflow.com/a/14038590

// Call this macro just after a kernel call to check for invalid launch arguments
#define cudaErrchkKernel()			{ cudaErrchkAPI( cudaPeekAtLastError() ); }
// Call cudaErrchkKernelAndSync to also check for execution errors
#define cudaErrchkKernelAndSync()	{ cudaErrchkAPI( cudaPeekAtLastError() ); cudaErrchkAPI( cudaDeviceSynchronize() ); }
// Wrap any CUDA API calls with cudaErrchkAPI to check for any errors
#define cudaErrchkAPI(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // CUDA_ERR_HANDLING

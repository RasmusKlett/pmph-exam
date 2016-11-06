#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"
#include "kernels.cu"
#include "ProjHelperFun.cu"
#include "cudaErrHandling.cu"
#include <vector>

// void printArray(vector<REAL> arr) {
//     printf("[");
//     for (const auto& elem : arr) {
//         printf("%f, ", elem);
//     }
//     printf("]\n");
// }

void
rollback(
                const unsigned g,
                PrivGlobs& globs,
                REAL* d_myResult,
                REAL* d_myVarX,
                REAL* d_myVarY,
                REAL* d_myDxx,
                REAL* d_myDyy,
                REAL* d_u,
                REAL* d_v,
                REAL* d_c,
                REAL* d_yy,
                const unsigned int outer,
                const unsigned dim
) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    /*      Call kernel  */
    const bool is_3D = outer * numX < 5000;

    int dimO = ceil( ((float)outer) / (is_3D ? 16 : dim ));
    int dimX = ceil( ((float)numX) / (is_3D ? 8 : dim ));
    int dimY = ceil( ((float)numY) / (is_3D ? 8 : dim ));

    dim3 block(dim, dim, 1), gridOX(dimO, dimX, 1);
    dim3 gridOY(dimO, dimY, 1);
    if (is_3D) {
        dim3 block3D(16, 8, 8), gridOXY(dimO, dimX, dimY);
        initUAndV3Dim<<<gridOXY, block3D>>>
            (d_u, d_v, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_myResult,
             outer, numX, numY, dtInv);
    } else {
        initUAndV2Dim<<<gridOX, block>>>
            (d_u, d_v, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_myResult,
             outer, numX, numY, dtInv);
    }

    tridag1<<<gridOY, block>>>(outer, numX, numY, numZ, d_c, dtInv, d_myVarX, d_myDxx, d_u, d_yy);
    tridag2<<<gridOX, block>>>(outer, numX, numY, numZ, d_c, dtInv, d_myVarY, d_myDyy, d_u, d_v, d_yy, d_myResult);
}

struct d_alloc {
    REAL** ptr;
    unsigned int size;
};

void   run_OrigCPU(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {

    PrivGlobs    globs(numX, numY, numT);
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);
    unsigned int numZ = max(numX, numY);

    /*      Declare device pointers */
    REAL *d_myResult, *d_myX, *d_myY, *d_myVarX, *d_myVarY, *d_myDxx, *d_myDyy, *d_u, *d_v, *d_res;
    REAL *d_c;
    REAL *d_yy;

    unsigned int myResultSize = outer * numX * numY;

    // List of device pointers and what size they request
    vector<d_alloc> allocations = {
        {&d_res, outer},
        {&d_myX, numX},
        {&d_myY, numY},
        {&d_myVarX, numX * numY},
        {&d_myVarY, numX * numY},
        {&d_myDxx, numX * 4},
        {&d_myDyy, numY * 4},
        {&d_c, outer*numZ*numZ},
        {&d_yy, outer * numZ * numZ},
        {&d_v, myResultSize},
        {&d_myResult, myResultSize},
        {&d_u, myResultSize},
    };

    unsigned long total_size;
    for (auto& al : allocations) {
        total_size += al.size;
    }

    // Allocate all device memory to save allocation/deallocation overhead
    // Unfortunately it doesn't seem to make too much difference.
    REAL* master_device_ptr;
    cudaErrchkAPI(cudaMalloc((void**)&master_device_ptr, total_size * sizeof(REAL)));

    REAL* accum_ptr = master_device_ptr;

    for (auto& al : allocations) {
        *(al.ptr) = accum_ptr;
        accum_ptr += al.size;
    }


    /* Copy initial required data to device */
    copy2DVec(d_myDxx, globs.myDxx, cudaMemcpyHostToDevice);
    copy2DVec(d_myDyy, globs.myDyy, cudaMemcpyHostToDevice);
    cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));
    cudaErrchkAPI(cudaMemcpy(d_myY, globs.myY.data(), numY * sizeof(REAL), cudaMemcpyHostToDevice));

    /* Compute myResult from a 2d kernel */
    int dim;
    if (outer > 31) {
        dim = 32;
    } else {
        dim = 16;
    }
    int dimO = ceil(((float)outer) / dim);
    int dimX = ceil(((float)numX) / dim);
    int dimY = ceil(((float)numY) / dim);
    dim3 block(dim, dim, 1), gridOX(dimO, dimX, 1), gridXY(dimX, dimY, 1);

    myResultKernel2D<<<gridOX, block>>>(outer, numX, numY, d_myX, d_myResult);
    cudaErrchkKernelAndSync();

    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        {
            REAL nu2t = 0.5 * nu * nu * globs.myTimeline[g];
            myVarXYKernel<<<gridXY, block>>>(numX, numY, beta, nu2t, alpha, d_myX, d_myY, d_myVarX, d_myVarY);
            cudaErrchkKernelAndSync();
        }
        rollback(g, globs, d_myResult, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_u, d_v, d_c, d_yy, outer, dim);
    }

    {

        unsigned int block_size = 256;
        unsigned int num_blocks = (outer + (block_size -1)) / block_size;
        
        buildResultKernel<<<num_blocks, block_size>>>(outer, numX, numY, globs.myXindex, globs.myYindex, d_res, d_myResult);
        cudaErrchkKernelAndSync();

        cudaMemcpy(res, d_res, outer * sizeof(REAL), cudaMemcpyDeviceToHost);

    }

    cudaErrchkAPI(cudaFree(master_device_ptr));
}

//#endif // PROJ_CORE_ORIG

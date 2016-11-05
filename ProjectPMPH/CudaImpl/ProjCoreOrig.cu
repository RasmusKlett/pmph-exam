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
                REAL* d_a,
                REAL* d_b,
                REAL* d_c,
                REAL* d_yy,
                REAL* d__y,
                const unsigned int outer
) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    /*      Call kernel  */
    unsigned dim = 32;
    int dimO = ceil( ((float)outer) / dim );
    int dimX = ceil( ((float)numX) / dim );
    int dimY = ceil( ((float)numY) / dim );

    dim3 block(dim, dim, 1), gridOX(dimO, dimX, 1);
    dim3 gridOY(dimO, dimY, 1);

    initUAndV2Dim<<<gridOX, block>>>(d_u, d_v, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_myResult, outer, numX, numY, dtInv);
    tridag1<<<gridOY, block>>>(outer, numX, numY, numZ, d_a, d_b, d_c, dtInv, d_myVarX, d_myDxx, d_u, d_yy);
    tridag2<<<gridOX, block>>>(outer, numX, numY, numZ, d_a, d_b, d_c, dtInv, d_myVarY, d_myDyy, d_u, d_v, d_yy, d__y, d_myResult);
}

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
    int numZ = max(numX, numY);

    /*      Allocate Device memory */
    REAL *d_myResult, *d_myX, *d_myY, *d_myVarX, *d_myVarY, *d_myDxx, *d_myDyy, *d_u, *d_v;
    cudaErrchkAPI(cudaMalloc((void**)&d_myX, numX * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myY, numY * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myVarX, numX * numY * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myVarY, numX * numY * sizeof(REAL)));
    unsigned long myDxxSize = numX * 4 * sizeof(REAL);
    unsigned long myDyySize = numY * 4 * sizeof(REAL);
    cudaErrchkAPI(cudaMalloc((void**)&d_myDxx, myDxxSize));
    cudaErrchkAPI(cudaMalloc((void**)&d_myDyy, myDyySize));
    unsigned long myResultSize = outer * numX * numY * sizeof(REAL);
    cudaErrchkAPI(cudaMalloc((void**)&d_u, myResultSize));
    cudaErrchkAPI(cudaMalloc((void**)&d_v, myResultSize));
    cudaErrchkAPI(cudaMalloc((void**)&d_myResult, myResultSize));

    REAL *d_a, *d_b, *d_c;
    cudaErrchkAPI(cudaMalloc((void**)&d_a, outer * numZ * numZ * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_b, outer * numZ * numZ * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_c, outer * numZ * numZ * sizeof(REAL)));
    REAL *d_yy;
    cudaErrchkAPI(cudaMalloc((void**)&d_yy, outer * numZ * numZ * sizeof(REAL)));
    REAL *d__y;
    cudaErrchkAPI(cudaMalloc((void**)&d__y, outer * numZ * numZ * sizeof(REAL)));

    /* Copy initial required data to device */
    copy2DVec(d_myDxx, globs.myDxx, cudaMemcpyHostToDevice);
    copy2DVec(d_myDyy, globs.myDyy, cudaMemcpyHostToDevice);
    cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));
    cudaErrchkAPI(cudaMemcpy(d_myY, globs.myY.data(), numY * sizeof(REAL), cudaMemcpyHostToDevice));

    /* Compute myResult from a 2d kernel */
    int dim = 32;
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
        rollback(g, globs, d_myResult, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_u, d_v, d_a, d_b, d_c, d_yy, d__y, outer);
    }

    {
        REAL *d_res;
        cudaErrchkAPI(cudaMalloc((void**)&d_res, outer * sizeof(REAL)));

        unsigned int block_size = 256;
        unsigned int num_blocks = (outer + (block_size -1)) / block_size;
        
        buildResultKernel<<<num_blocks, block_size>>>(outer, numX, numY, globs.myXindex, globs.myYindex, d_res, d_myResult);
        cudaErrchkKernelAndSync();

        cudaMemcpy(res, d_res, outer * sizeof(REAL), cudaMemcpyDeviceToHost);

        cudaErrchkAPI(cudaFree(d_res));
    }

    cudaErrchkAPI(cudaFree(d_myX));
    cudaErrchkAPI(cudaFree(d_myY));
    cudaErrchkAPI(cudaFree(d_myVarX));
    cudaErrchkAPI(cudaFree(d_myVarY));
    cudaErrchkAPI(cudaFree(d_myDxx));
    cudaErrchkAPI(cudaFree(d_myDyy));
    cudaErrchkAPI(cudaFree(d_u));
    cudaErrchkAPI(cudaFree(d_v));

    cudaErrchkAPI(cudaFree(d_a));
    cudaErrchkAPI(cudaFree(d_b));
    cudaErrchkAPI(cudaFree(d_c));
    cudaErrchkAPI(cudaFree(d_yy));
    cudaErrchkAPI(cudaFree(d__y));
}

//#endif // PROJ_CORE_ORIG

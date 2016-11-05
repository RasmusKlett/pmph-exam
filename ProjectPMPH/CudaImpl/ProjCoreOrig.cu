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
                vector<vector<vector<REAL > > >& myResult,
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
                const unsigned int outer
) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    unsigned x, y;
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<vector<REAL> > > u(outer, vector<vector<REAL> > (numY, vector<REAL>(numX))); // [outer][numY][numX]
    vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY))); // [outer][numX][numY]

    vector<vector<vector<REAL> > > a(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > b(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > c(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]

    vector<vector<vector<REAL> > > _y(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > yy(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // temporary used in tridag  // [outer][numZ][numZ]


    /*      Copy data to device  */
    copy3DVec(d_myResult, myResult, cudaMemcpyHostToDevice);

    /*      Call kernel  */
    unsigned dim = 32;
    int dimO = ceil( ((float)outer) / dim );
    int dimX = ceil( ((float)numX) / dim );
    int dimY = ceil( ((float)numY) / dim );

    dim3 block(dim, dim, 1), grid(dimO, dimX, 1);
    dim3 gridOY(dimO, dimY, 1);

    initUAndV2Dim<<<grid, block>>>(d_u, d_v, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_myResult, outer, numX, numY, dtInv);
    tridag1<<<gridOY, block>>>(
            outer, numX, numY, numZ,
            d_a, d_b, d_c,
            dtInv,
            d_myVarX, d_myDxx,
            d_u, d_yy
            );

    /*      Copy data back to host */
    copy3DVec(d_v, v, cudaMemcpyDeviceToHost);
    copy3DVec(d_u, u, cudaMemcpyDeviceToHost);

    for( unsigned o = 0; o < outer; ++ o ) {
        //  implicit y
        for(x = 0; x < numX; x++) {
            for(y = 0; y < numY; y++) {  // here a, b, c should have size [numY]
                a[o][x][y] =       - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][0]);
                b[o][x][y] = dtInv - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][1]);
                c[o][x][y] =       - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][2]);
            }

            for(y = 0; y < numY; y++) {
                _y[o][x][y] = dtInv*u[o][y][x] - 0.5*v[o][x][y];
            }

            // here yy should have size [numY]
            tridagPar(a[o][x],b[o][x],c[o][x],_y[o][x],numY,myResult[o][x],yy[o][x]);
        }
    }
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

    vector<vector<vector<REAL> > > myResult(outer, vector<vector<REAL > >(numX, vector<REAL> (numY)));

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

    copy2DVec(d_myDxx, globs.myDxx, cudaMemcpyHostToDevice);
    copy2DVec(d_myDyy, globs.myDyy, cudaMemcpyHostToDevice);

    // Compute myResult from a 2d kernel
    int dim = 32;
    int dimO = ceil(((float)outer) / dim);
    int dimX = ceil(((float)numX) / dim);
    int dimY = ceil(((float)numY) / dim);
    dim3 block(dim, dim, 1), gridOX(dimO, dimX, 1), gridXY(dimX, dimY, 1);

    cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));
    cudaErrchkAPI(cudaMemcpy(d_myY, globs.myY.data(), numY * sizeof(REAL), cudaMemcpyHostToDevice));

    myResultKernel2D<<<gridOX, block>>>(outer, numX, numY, d_myX, d_myResult);
    cudaErrchkKernelAndSync();

    copy3DVec(d_myResult, myResult, cudaMemcpyDeviceToHost);

    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        {
            REAL nu2t = 0.5 * nu * nu * globs.myTimeline[g];
            myVarXYKernel<<<gridXY, block>>>(numX, numY, beta, nu2t, alpha, d_myX, d_myY, d_myVarX, d_myVarY);
            cudaErrchkKernelAndSync();

            copy2DVec(d_myVarX, globs.myVarX, cudaMemcpyDeviceToHost);
            copy2DVec(d_myVarY, globs.myVarY, cudaMemcpyDeviceToHost);
        }
        rollback(g, globs, myResult, d_myResult, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_u, d_v, d_a, d_b, d_c, d_yy, outer);
    }

    cudaErrchkAPI(cudaFree(d_myX));
    cudaErrchkAPI(cudaFree(d_myY));
    cudaErrchkAPI(cudaFree(d_myVarX));
    cudaErrchkAPI(cudaFree(d_myVarY));
    cudaErrchkAPI(cudaFree(d_myDxx));
    cudaErrchkAPI(cudaFree(d_myDyy));
    cudaErrchkAPI(cudaFree(d_u));
    cudaErrchkAPI(cudaFree(d_v));
    cudaErrchkAPI(cudaFree(d_myResult));

    cudaErrchkAPI(cudaFree(d_a));
    cudaErrchkAPI(cudaFree(d_b));
    cudaErrchkAPI(cudaFree(d_c));
    cudaErrchkAPI(cudaFree(d_yy));


    {
        REAL *d_myResult;
        REAL *d_res;
        cudaErrchkAPI(cudaMalloc((void**)&d_myResult, outer * numX * numY * sizeof(REAL)));
        cudaErrchkAPI(cudaMalloc((void**)&d_res, outer * sizeof(REAL)));

        copy3DVec(d_myResult, myResult, cudaMemcpyHostToDevice);

        unsigned int block_size = 256;
        unsigned int num_blocks = (outer + (block_size -1)) / block_size;
        
        buildResultKernel<<<num_blocks, block_size>>>(outer, numX, numY, globs.myXindex, globs.myYindex, d_res, d_myResult);
        cudaErrchkKernelAndSync();

        cudaMemcpy(res, d_res, outer * sizeof(REAL), cudaMemcpyDeviceToHost);

        cudaErrchkAPI(cudaFree(d_myResult));
        cudaErrchkAPI(cudaFree(d_res));
    }
}

//#endif // PROJ_CORE_ORIG

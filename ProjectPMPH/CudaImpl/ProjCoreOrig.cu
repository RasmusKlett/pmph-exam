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
rollback( const unsigned g, PrivGlobs& globs, vector<vector<vector<REAL > > >& myResult, REAL* d_myResult, const unsigned int outer) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    unsigned x, y, i, j;
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<vector<REAL> > > u(outer, vector<vector<REAL> > (numY, vector<REAL>(numX))); // [outer][numY][numX]
    vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<REAL> a(numZ); // [max(numX,numY)]
    vector<REAL> b(numZ); // [max(numX,numY)]
    vector<REAL> c(numZ); // [max(numX,numY)]
    vector<vector<REAL> > _y(outer, vector<REAL>(numZ)); // [outer][max(numX,numY)]

    vector<vector<REAL> > yy(outer, vector<REAL>(numZ));  // temporary used in tridag  // [max(numX,numY)]

    REAL* d_myVarX;
    REAL* d_myVarY;
    REAL* d_myDxx;
    REAL* d_myDyy;
    REAL* d_u;
    REAL* d_v;


    /*      Allocate device memory  */
    unsigned long myVarSize = numX * numY * sizeof(REAL);
    cudaMalloc((void**)&d_myVarX, myVarSize);
    cudaMalloc((void**)&d_myVarY, myVarSize);
    unsigned long myDxxSize = numX * 4 * sizeof(REAL);
    unsigned long myDyySize = numY * 4 * sizeof(REAL);
    cudaMalloc((void**)&d_myDxx, myDxxSize);
    cudaMalloc((void**)&d_myDyy, myDyySize);
    unsigned long myResultSize = outer * numX * numY * sizeof(REAL);
    cudaMalloc((void**)&d_u, myResultSize);
    cudaMalloc((void**)&d_v, myResultSize);

    /*      Copy data to device  */
    copy2DVec(d_myVarX, globs.myVarX, cudaMemcpyHostToDevice);
    copy2DVec(d_myVarY, globs.myVarY, cudaMemcpyHostToDevice);
    copy2DVec(d_myDxx, globs.myDxx, cudaMemcpyHostToDevice);
    copy2DVec(d_myDyy, globs.myDyy, cudaMemcpyHostToDevice);
    copy3DVec(d_myResult, myResult, cudaMemcpyHostToDevice);

    /*      Call kernel  */
    unsigned dim = 32;
    int dimO = ceil( ((float)outer) / dim );
    int dimX = ceil( ((float)numX) / dim );

    dim3 block(dim, dim, 1), grid(dimO, dimX, 1);

    initUAndV2Dim<<<grid, block>>>(d_u, d_v, d_myVarX, d_myVarY, d_myDxx, d_myDyy, d_myResult, outer, numX, numY, dtInv);

    /*      Copy data back to host */
    copy3DVec(d_u, u, cudaMemcpyDeviceToHost);
    copy3DVec(d_v, v, cudaMemcpyDeviceToHost);


    /*      Free device memory  */
    cudaFree(d_myVarX);
    cudaFree(d_myVarY);
    cudaFree(d_myDxx);
    cudaFree(d_myDyy);
    cudaFree(d_u);
    cudaFree(d_v);

    for( unsigned o = 0; o < outer; ++ o ) {
        //  implicit x
        for(y = 0; y < numY; y++) {
            for(x = 0; x < numX; x++) {  // here a, b,c should have size [numX]
                a[x] =       - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][0]);
                b[x] = dtInv - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][1]);
                c[x] =       - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][2]);
            }
            // here yy should have size [numX]
            tridagPar(a,b,c,u[o][y],numX,u[o][y],yy[o]);
        }

        //  implicit y
        for(x = 0; x < numX; x++) {
            for(y = 0; y < numY; y++) {  // here a, b, c should have size [numY]
                a[y] =       - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][0]);
                b[y] = dtInv - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][1]);
                c[y] =       - 0.5*(0.5*globs.myVarY[x][y]*globs.myDyy[y][2]);
            }

            for(y = 0; y < numY; y++) {
                _y[o][y] = dtInv*u[o][y][x] - 0.5*v[o][x][y];
            }

            // here yy should have size [numY]
            tridagPar(a,b,c,_y[o],numY,myResult[o][x],yy[o]);
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

    vector<vector<vector<REAL> > > myResult(outer, vector<vector<REAL > >(numX, vector<REAL> (numY)));

    REAL *d_myResult;
    // Compute myResult from a 2d kernel
    {
        cudaErrchkAPI(cudaMalloc((void**)&d_myResult, outer * numX * numY * sizeof(REAL)));

        REAL *d_myX;
        cudaErrchkAPI(cudaMalloc((void**)&d_myX, numX * sizeof(REAL)));
        cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));

        int T =32;
        int dimy = ceil(((float)outer) / T);
        int dimx = ceil(((float)numX) / T);
        dim3 block(T, T, 1), grid(dimx, dimy, 1);

        myResultKernel2D<<<grid, block>>>(outer, numX, numY, d_myX, d_myResult);
        cudaErrchkKernelAndSync();

        copy3DVec(d_myResult, myResult, cudaMemcpyDeviceToHost);

        cudaErrchkAPI(cudaFree(d_myX));
    }

    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        for(unsigned x = 0; x < globs.myX.size(); ++x) {
            for(unsigned y = 0; y < globs.myY.size(); ++y) {
                globs.myVarX[x][y] = exp(2.0*(  beta*log(globs.myX[x])
                                              + globs.myY[y]
                                              - 0.5*nu*nu*globs.myTimeline[g] )
                                        );
                globs.myVarY[x][y] = exp(2.0*(  alpha*log(globs.myX[x])
                                              + globs.myY[y]
                                              - 0.5*nu*nu*globs.myTimeline[g] )
                                        ); // nu*nu
            }
        }

        rollback(g, globs, myResult, d_myResult, outer);
        // g = -1;
    }
    cudaFree(d_myResult);
    for( unsigned o = 0; o < outer; ++o ) {
        res[o] = myResult[o][globs.myXindex][globs.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG

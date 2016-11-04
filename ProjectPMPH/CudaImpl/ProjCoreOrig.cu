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

void rollbackUandV(const unsigned int outer, unsigned int numX) {
    int dimy = outer;//ceil(((float)outer) );
    int dimx = numX;//ceil(((float)numX) / T);
    dim3 block(32, 32, 1), grid(dimx, dimy, 1);


    printf("\nprinting just before kernel\n");
    initUAndV2Dim<<<grid, block>>>();

}

void
rollback( const unsigned g, PrivGlobs& globs, vector<vector<vector<REAL > > >& myResult, const unsigned int outer) {
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

rollbackUandV(outer, numX);
    for( unsigned o = 0; o < outer; ++ o )
    {
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
                // explicit x
                u[o][j][i] = dtInv*myResult[o][i][j];

                if(i > 0) {
                  u[o][j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] )
                                * myResult[o][i-1][j];
                }
                u[o][j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                                * myResult[o][i][j];
                if(i < numX-1) {
                  u[o][j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                                * myResult[o][i+1][j];
                }

                // explicit y
                v[o][i][j] = 0.0;

                if(j > 0) {
                  v[o][i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                             *  myResult[o][i][j-1];
                }
                v[o][i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                             *  myResult[o][i][j];
                if(j < numY-1) {
                  v[o][i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                             *  myResult[o][i][j+1];
                }
                u[o][j][i] += v[o][i][j];
            }
        }

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

    // Compute myResult from a 2d kernel
    {
        REAL *d_myResult;
        cudaErrchkAPI(cudaMalloc((void**)&d_myResult, outer * numX * numY * sizeof(REAL)));

        REAL *d_myX;
        cudaErrchkAPI(cudaMalloc((void**)&d_myX, numX * sizeof(REAL)));
        cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));

        int T =32;
        int dimx = ceil(((float)outer) / T);
        int dimy = ceil(((float)numX) / T);
        dim3 block(T, T, 1), grid(dimx, dimy, 1);

        myResultKernel2D<<<grid, block>>>(outer, numX, numY, d_myX, d_myResult);
        cudaErrchkKernelAndSync();

        copy3DVec(d_myResult, myResult, cudaMemcpyDeviceToHost);

        cudaErrchkAPI(cudaFree(d_myResult));
        cudaErrchkAPI(cudaFree(d_myX));
    }

    REAL *d_myX, *d_myY, *d_myVarX, *d_myVarY;
    cudaErrchkAPI(cudaMalloc((void**)&d_myX, numX * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myY, numY * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myVarX, numX * numY * sizeof(REAL)));
    cudaErrchkAPI(cudaMalloc((void**)&d_myVarY, numX * numY * sizeof(REAL)));

    cudaErrchkAPI(cudaMemcpy(d_myX, globs.myX.data(), numX * sizeof(REAL), cudaMemcpyHostToDevice));
    cudaErrchkAPI(cudaMemcpy(d_myY, globs.myY.data(), numY * sizeof(REAL), cudaMemcpyHostToDevice));

    int T =32;
    int dimx = ceil(((float)numX) / T);
    int dimy = ceil(((float)numY) / T);
    dim3 block(T, T, 1), grid(dimx, dimy, 1);


    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        {
            REAL nu2t = 0.5 * nu * nu * globs.myTimeline[g];
            myVarXYKernel<<<grid, block>>>(numX, numY, beta, nu2t, alpha, d_myX, d_myY, d_myVarX, d_myVarY);
            cudaErrchkKernelAndSync();

            copy2DVec(d_myVarX, globs.myVarX, cudaMemcpyDeviceToHost);
            copy2DVec(d_myVarY, globs.myVarY, cudaMemcpyDeviceToHost);
        }
        rollback(g, globs, myResult, outer);
    }

    cudaErrchkAPI(cudaFree(d_myX));
    cudaErrchkAPI(cudaFree(d_myY));
    cudaErrchkAPI(cudaFree(d_myVarX));
    cudaErrchkAPI(cudaFree(d_myVarY));


    for( unsigned o = 0; o < outer; ++o ) {
        res[o] = myResult[o][globs.myXindex][globs.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG

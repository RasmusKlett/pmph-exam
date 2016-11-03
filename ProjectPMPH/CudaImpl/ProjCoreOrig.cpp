#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"
#include <vector>

void printArray(vector<REAL> arr) {
    printf("[");
    for (const auto& elem : arr) {
        printf("%f, ", elem);
    }
    printf("]\n");
}

void
rollback( const unsigned g, PrivGlobs& globs, vector<vector<vector<REAL > > >& myResult, const unsigned int outer) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    unsigned i, j;
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<vector<REAL> > > u(outer, vector<vector<REAL> > (numY, vector<REAL>(numX))); // [outer][numY][numX]
    vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<REAL> > a(outer, vector<REAL>(numZ)); // [outer][max(numX,numY)]
    vector<vector<REAL> > b(outer, vector<REAL>(numZ)); // [outer][max(numX,numY)]
    vector<vector<REAL> > c(outer, vector<REAL>(numZ)); // [outer][max(numX,numY)]
    vector<vector<REAL> > y(outer, vector<REAL>(numZ)); // [outer][max(numX,numY)]

    vector<vector<REAL> > yy(outer, vector<REAL>(numZ));  // temporary used in tridag  // [max(numX,numY)]

    for( unsigned o = 0; o < outer; ++ o )
    {
        //  explicit x
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
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
            }
        }

        //  explicit y
        for(j=0;j<numY;j++) {
            for(i=0;i<numX;i++) {
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
        // Privatize a,b,c (size will be (outer*?)*numY*numX)
        for(j=0;j<numY;j++) {
            for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                a[o][i] =       - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
                b[o][i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
                c[o][i] =       - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
            }
            // here yy should have size [numX]
            tridagPar(a[o],b[o],c[o],u[o][j],numX,u[o][j],yy[o]);
        }

        //  implicit y
        // TODO reuse arrays for abc from above
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                a[o][j] =       - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
                b[o][j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
                c[o][j] =       - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
            }

            for(j=0;j<numY;j++) {
                y[o][j] = dtInv*u[o][j][i] - 0.5*v[o][i][j];
            }

            // here yy should have size [numY]
            tridagPar(a[o],b[o],c[o],y[o],numY,myResult[o][i],yy[o]);
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

    for( unsigned ir = 0; ir < outer; ++ ir ) {
        for(unsigned i=0;i<globs.myX.size();++i) {
            for(unsigned j=0;j<globs.myY.size();++j) {
                myResult[ir][i][j] = max(globs.myX[i]-(0.001*ir), (REAL)0.0);;
            }
        }
    }

    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        for(unsigned i=0;i<globs.myX.size();++i) {
            for(unsigned j=0;j<globs.myY.size();++j) {
                globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])
                                              + globs.myY[j]
                                              - 0.5*nu*nu*globs.myTimeline[g] )
                                        );
                globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])
                                              + globs.myY[j]
                                              - 0.5*nu*nu*globs.myTimeline[g] )
                                        ); // nu*nu
            }
        }

        rollback(g, globs, myResult, outer);
    }
    for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = myResult[i][globs.myXindex][globs.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG

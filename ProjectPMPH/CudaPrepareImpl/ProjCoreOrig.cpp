#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"
#include <vector>

void
rollback( const unsigned g, PrivGlobs& globs, vector<vector<vector<REAL > > >& myResult, const unsigned int outer) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();
    unsigned numZ = max(numX,numY);
    unsigned o, x, y;
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<vector<REAL> > > u(outer, vector<vector<REAL> > (numY, vector<REAL>(numX))); // [outer][numY][numX]
    vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<vector<REAL> > > a(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > b(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > c(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > _y(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // [outer][numZ][numZ]
    vector<vector<vector<REAL> > > yy(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ))); // temporary used in tridag // [outer][numZ][numZ]

    for(o = 0; o < outer; o++) {
        for(x = 0; x < numX; x ++) {
            for(y = 0; y < numY; y++) {
                // explicit x
                REAL u_new = dtInv*myResult[o][x][y];

                if(x > 0) {
                  u_new += 0.5*( 0.5*globs.myVarX[x][y]*globs.myDxx[x][0] )
                            * myResult[o][x-1][y];
                }
                u_new  +=  0.5*( 0.5*globs.myVarX[x][y]*globs.myDxx[x][1] )
                            * myResult[o][x][y];
                if(x < numX-1) {
                  u_new += 0.5*( 0.5*globs.myVarX[x][y]*globs.myDxx[x][2] )
                            * myResult[o][x+1][y];
                }

                // explicit y
                REAL v_new = 0.0;

                if(y > 0) {
                  v_new +=  ( 0.5*globs.myVarY[x][y]*globs.myDyy[y][0] )
                         *  myResult[o][x][y-1];
                }
                v_new  +=   ( 0.5*globs.myVarY[x][y]*globs.myDyy[y][1] )
                         *  myResult[o][x][y];
                if(y < numY-1) {
                  v_new +=  ( 0.5*globs.myVarY[x][y]*globs.myDyy[y][2] )
                         *  myResult[o][x][y+1];
                }
                v[o][x][y] = v_new;
                u[o][y][x] = u_new + v_new;
            }
        }
    }

    for(o = 0; o < outer; o++) {
        //  implicit x
        for(y = 0; y < numY; y++) {
            for(x = 0; x < numX; x++) {  // here a, b,c should have size [numX]
                a[o][y][x] =       - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][0]);
                b[o][y][x] = dtInv - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][1]);
                c[o][y][x] =       - 0.5*(0.5*globs.myVarX[x][y]*globs.myDxx[x][2]);
            }
            // here yy should have size [numX]
            tridagPar(a[o][y],b[o][y],c[o][y],u[o][y],numX,u[o][y],yy[o][y]);
        }
    }
    
    for(o = 0; o < outer; o++) {
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

    vector<vector<vector<REAL> > > myResult(outer, vector<vector<REAL > >(numX, vector<REAL> (numY)));

    for( unsigned o = 0; o < outer; ++ o ) {
        for(unsigned x = 0; x < globs.myX.size(); ++x) {
            REAL v = max(globs.myX[x]-(0.001*o), (REAL)0.0);
            for(unsigned y = 0; y < globs.myY.size(); ++y) {
                myResult[o][x][y] = v;
            }
        }
    }

    for(int g = globs.myTimeline.size()-2;g>=0;--g) {
        REAL nu2t = 0.5*nu*nu*globs.myTimeline[g];
        for(unsigned x = 0; x < globs.myX.size(); ++x) {
            for(unsigned y = 0; y < globs.myY.size(); ++y) {
                globs.myVarX[x][y] = exp(2.0*(  beta*log(globs.myX[x])
                                              + globs.myY[y]
                                              - nu2t)
                                        );
                globs.myVarY[x][y] = exp(2.0*(  alpha*log(globs.myX[x])
                                              + globs.myY[y]
                                              - nu2t)
                                        ); // nu*nu
            }
        }

        rollback(g, globs, myResult, outer);
    }
    for( unsigned o = 0; o < outer; ++o ) {
        res[o] = myResult[o][globs.myXindex][globs.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG

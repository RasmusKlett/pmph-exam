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

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
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
}

void setPayoff(const REAL strike, PrivGlobs& globs, vector<vector<REAL > >& myResult)
{
	for(unsigned i=0;i<globs.myX.size();++i) {
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j) {
            //printf("%i,%i: %f\n", i, j, payoff);
			myResult[i][j] = payoff;
        }
	}
}

void
rollback( const unsigned g, PrivGlobs& globs, vector<vector<REAL> >& myResult) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*myResult[i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] ) 
                            * myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * myResult[i+1][j];
            }
        }
    }

    //	explicit y
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  myResult[i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            c[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            c[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
        }

        for(j=0;j<numY;j++) {
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];
        }

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,myResult[i],yy);
    }
}

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT,
                vector<vector<REAL > >& myResult
) {	

    for(int i = globs.myTimeline.size()-2;i>=0;--i) {
        updateParams(i,alpha,beta,nu,globs);
        rollback(i, globs, myResult);
    }

    return myResult[globs.myXindex][globs.myYindex];
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

    // printArray(myResult[1][1]);
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike = 0.001*i;
        setPayoff(strike, globs, myResult[i]);
    }

    for(int j = globs.myTimeline.size()-2;j>=0;--j) {
        for( unsigned i = 0; i < outer; ++ i ) {
            updateParams(j,alpha,beta,nu,globs);
            rollback(j, globs, myResult[i]);
        }

        // res[i] = myResult[i][globs.myXindex][globs.myYindex];

        /* Original 
        res[i] = value( globs, s0, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT, myResult[i] );
        */
    }
    for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = myResult[i][globs.myXindex][globs.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG

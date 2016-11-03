#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    for(unsigned i=0;i<globs.myX.size();++i)
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

void setPayoff(const REAL strike, PrivGlobs& globs )
{
	for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff;
	}
}


void
rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]

    vector<vector<REAL> > aX(numZ, vector<REAL>(numZ));
    vector<vector<REAL> > bX(numZ, vector<REAL>(numZ));
    vector<vector<REAL> > cX(numZ, vector<REAL>(numZ));

    vector<vector<REAL> > aY(numZ, vector<REAL>(numZ));
    vector<vector<REAL> > bY(numZ, vector<REAL>(numZ));
    vector<vector<REAL> > cY(numZ, vector<REAL>(numZ));

    vector<REAL> y(numZ);
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            //	explicit x
            u[j][i] = dtInv*globs.myResult[i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] ) 
                            * globs.myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * globs.myResult[i+1][j];
            }

            //	explicit y
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  globs.myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  globs.myResult[i][j+1];
            }
            u[j][i] += v[i][j]; 

            // implicit x
            aX[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            bX[j][i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            cX[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);

            // implicit y
            aY[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            bY[i][j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            cY[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        // here yy should have size [numX]
        tridagPar(aX[j],bX[j],cX[j],u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        //
        tridagPar(aY[i],bY[i],cY[i],y,numY,globs.myResult[i],yy);
    }
}

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {	
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);

    setPayoff(strike, globs);
    for(int i = globs.myTimeline.size()-2;i>=0;--i)
    {
        updateParams(i,alpha,beta,nu,globs);
        rollback(i, globs);
    }

    return globs.myResult[globs.myXindex][globs.myYindex];
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

    #pragma omp parallel for default(shared) schedule(static)
        for( unsigned i = 0; i < outer; ++ i ) {
            REAL strike;
            PrivGlobs    globs(numX, numY, numT);
            strike = 0.001*i;
            res[i] = value( globs, s0, strike, t,
                            alpha, nu,    beta,
                            numX,  numY,  numT );
        }
}

//#endif // PROJ_CORE_ORIG
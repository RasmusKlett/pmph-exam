ProjectMain.cpp:int main()
{
    /* ... */
    ParseInput.h:readDataSet( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T ); 
    /* ... */

    {   // Original Program (Sequential CPU Execution)

        ProjCoreOrig.cpp:run_OrigCPU( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res )
        {
            /* ... */
            for(/* ... */) {
                /* ... */
    /* ... */ = ProjCoreOrig.cpp:value(globs, s0, strike, t, alpha, nu, beta, numX, numY, numT)
                {
                    ProjHelperFun.cpp:initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
                    ProjHelperFun.cpp:initOperator(globs.myX,globs.myDxx);
                    ProjHelperFun.cpp:initOperator(globs.myY,globs.myDyy);

                    ProjCoreOrig.cpp:setPayoff(strike, globs);

                    for(/* ... */)
                    {
                        ProjCoreOrig.cpp:updateParams(i,alpha,beta,nu,globs);
                        ProjCoreOrig.cpp:rollback(i, globs)
                        {
                            /* ... */
                            for(/* ... */) {
                                /* ... */
                                TridagPar.h:tridagPar(a,b,c,u[j],numX,u[j],yy);
                            }

                            for(/* ... */) {
                                /* ... */
                                TridagPar.h:tridagPar(a,b,c,y,numY,globs.myResult[i],yy);
                            }
                        }
                    }
                    /* ... */
                }
            }
        }
        /* ... */
    }
    /* ... */
}


__device__ inline void initUAndV (
                    REAL* u,
                    REAL* v,
                    REAL* myVarX,
                    REAL* myVarY,
                    REAL* myDxx,
                    REAL* myDyy,
                    REAL* myResult,
                    const int outer,
                    const int numX,
                    const int numY,
                    REAL dtInv,
                    const int o,
                    const int x, 
                    const int y
) {
    REAL u_new = dtInv * myResult[(x*numY*outer) + (y*outer) + o];

    if (x > 0) {
        u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 0])
            * myResult[((x-1)*numY*outer) + (y*outer) + o];
    }
    u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 1])
        * myResult[(x*numY*outer) + (y*outer) + o];
    if (x < numX - 1) {
        u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 2])
            * myResult[((x+1)*numY*outer) + (y*outer) + o];
    }

    // explicit y
    REAL v_new = 0.0;
    if(y > 0) {
        v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 0])
            *  myResult[(x*numY*outer) + ((y-1)*outer) + o];
    }
    v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 1])
        *  myResult[(x*numY*outer) + (y*outer) + o];
    if(y < numY - 1) {
        v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 2])
            *  myResult[(x*numY*outer) + ((y+1)*outer) + o];
    }
    v[(x*numY*outer) + (y*outer) + o] = v_new;
    u[(y*numX*outer) + (x*outer) + o] = u_new + v_new;
}

__global__ void initUAndV3Dim (
                    REAL* u,
                    REAL* v,
                    REAL* myVarX,
                    REAL* myVarY,
                    REAL* myDxx,
                    REAL* myDyy,
                    REAL* myResult,
                    const int outer,
                    const int numX,
                    const int numY,
                    REAL dtInv
) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;
    int y = blockIdx.z*blockDim.z + threadIdx.z;
    if (o < outer && x < numX && y < numY) {
        initUAndV(
            u, v, myVarX, myVarY, myDxx, myDyy, myResult,
            outer, numX, numY, dtInv,
            o, x, y
        );
    }
}

__global__ void initUAndV2Dim (
                    REAL* u,
                    REAL* v,
                    REAL* myVarX,
                    REAL* myVarY,
                    REAL* myDxx,
                    REAL* myDyy,
                    REAL* myResult,
                    const int outer,
                    const int numX,
                    const int numY,
                    REAL dtInv
) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;

    if (o < outer && x < numX) {
        REAL myDxx0 = myDxx[(x*4) + 0];
        REAL myDxx1 = myDxx[(x*4) + 1];
        REAL myDxx2 = myDxx[(x*4) + 2];
        for(int y = 0; y < numY; y++) {

            // explicit x
            REAL u_new = dtInv * myResult[(x*numY*outer) + (y*outer) + o];
            REAL varX = 0.5*myVarX[(x*numY) + y];

            if (x > 0) {
                u_new += 0.5*(varX * myDxx0)
                            * myResult[((x-1)*numY*outer) + (y*outer) + o];
            }
            u_new += 0.5*( varX * myDxx1)
                        * myResult[(x*numY*outer) + (y*outer) + o];
            if (x < numX - 1) {
                u_new += 0.5*( varX * myDxx2)
                            * myResult[((x+1)*numY*outer) + (y*outer) + o];
            }

            // explicit y
            REAL v_new = 0.0;
            REAL varY = 0.5 * myVarY[(x*numY) + y];
            if(y > 0) {
                v_new += ( varY * myDyy[(y*4) + 0])
                    *  myResult[(x*numY*outer) + ((y-1)*outer) + o];
            }
            v_new += ( varY * myDyy[(y*4) + 1])
                *  myResult[(x*numY*outer) + (y*outer) + o];
            if(y < numY - 1) {
                v_new += ( varY * myDyy[(y*4) + 2])
                    *  myResult[(x*numY*outer) + ((y+1)*outer) + o];
            }
            v[(x*numY*outer) + (y*outer) + o] = v_new;
            u[(y*numX*outer) + (x*outer) + o] = u_new + v_new;
        }
    }
}

__global__ void myResultKernel2D(const int outer, const int numX, const int numY, REAL *myX, REAL *myResult) {
	int o = threadIdx.x + blockDim.x*blockIdx.x;
  	int x = threadIdx.y + blockDim.y*blockIdx.y;

  	if (o < outer && x < numX) {
  		REAL v = max(myX[x]-(0.001*o), (REAL)0.0);
        for(int y = 0; y < numY; y++) {
            myResult[(x*numY*outer) + (y*outer) + o] = v;
        }
	}
}

__global__ void myVarXYKernel(
	const int numX, const int numY,
	REAL beta, REAL nu2t, REAL alpha,
	REAL *myX, REAL *myY,
	REAL *myVarX, REAL *myVarY
	) {
	int x = threadIdx.x + blockDim.x*blockIdx.x;
  	int y = threadIdx.y + blockDim.y*blockIdx.y;

  	if (x < numX && y < numY) {
        myVarX[x * numY + y] = exp(2.0*(  beta*log(myX[x])
	                                      + myY[y]
	                                      - nu2t )
	                                );
        myVarY[x * numY + y] = exp(2.0*(  alpha*log(myX[x])
	                                      + myY[y]
	                                      - nu2t )
	                                ); // nu*nu
	}
}

__global__ void buildResultKernel(
	const int outer, const int numX, const int numY,
	const int myXindex, const int myYindex,
	REAL *res, REAL *myResult
	) {
	const int o = threadIdx.x + blockDim.x * blockIdx.x;

	if (o < outer) {
        res[o] = myResult[o * numX * numY + myXindex * numY + myYindex];
        res[o] = myResult[(myXindex*numY*outer) + (myYindex*outer) + o];
    }
}
// Sets up a, b, c and performs tridag. Calculation of a and b has been privatized, 
// and c has been moved.
__global__ void tridag1(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *c,
    REAL dtInv,
    REAL *myVarX, REAL *myDxx,
    REAL *u, REAL *yy
    ) {
    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && y < numY) {

        // Inlined tridag
        c = c + (o + y * outer);
        REAL* r = u + (o + y * numX * outer);
        u = u + (o + y * numX * outer);
        yy = yy + (o + y * outer);

        REAL   beta;

        u[0]  = r[0];
        yy[0] = dtInv - 0.5*(0.5*myVarX[y]*myDxx[1]);

        for(int x=1; x<numX; x++) {
            REAL aV =       - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 0]);
            REAL bV = dtInv - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 1]);
            REAL cV =       - 0.5*(0.5*myVarX[(x-1) * numY + y]*myDxx[(x-1) * 4 + 2]);

            // Write this in loop, it is not worth it to recalculate in following loop
            c[x * numZ * outer] = cV;

            beta  = aV / yy[(x-1) * numZ * outer];

            yy[x * numZ * outer] = bV - beta*cV;
            u[x * outer]  = r[x*outer] - beta*u[(x-1) * outer];
        }

        // X) this is a backward recurrence
        u[(numX-1)*outer] = u[(numX-1)*outer] / yy[(numX-1) * numZ * outer];
        for(int x=numX-2; x>=0; x--) {
            u[x*outer] = (u[x*outer] - c[(x+1) * numZ * outer]*u[(x+1)*outer]) / yy[x * numZ * outer];
        }
    }
}

// Sets up a, b, c, r and performs tridag. Calculation of a, b and r has
// been privatized, and c has been moved.
__global__ void tridag2(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *c,
    REAL dtInv,
    REAL *myVarY, REAL *myDyy,
    REAL *u, REAL *v, REAL *yy, REAL *myResult
    ) {

    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int x = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && x < numX) {

        // Inlined tridag
        REAL* u_old = u;
        u = myResult + (o + x * numY * outer);
        yy = yy + o + x * numZ * outer;

        REAL   beta;

        u[0]  = dtInv*u[(x*outer) + o] - 0.5*v[x * numY * outer + o];
        yy[0] = dtInv - 0.5*(0.5*myVarY[x * numY]*myDyy[1]);

        for(int y=1; y<numY; y++) {
            REAL aV = - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 0]);
            REAL bV = dtInv - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 1]);
            REAL cV =       - 0.5*(0.5*myVarY[x * numY + (y-1)]*myDyy[(y-1) * 4 + 2]);

            // Write this in loop, it is not worth it to recalculate in following loop
            c[(x*numZ*outer) + y * outer + o] = cV;

            REAL rV = dtInv*u_old[(y*numX*outer) + (x*outer) + o] - 0.5*v[x * numY * outer + y * outer + o];

            beta  = aV / yy[(y-1) * outer];

            yy[y * outer] = bV - beta*cV;
            u[y*outer]  = rV - beta*u[(y-1)*outer];
        }

        // X) this is a backward recurrence
        u[(numY-1)*outer] = u[(numY-1)*outer] / yy[(numY-1) * outer];
        for(int i=numY-2; i>=0; i--) {
            u[i*outer] = (u[i*outer] - c[(x*numZ*outer) + (i+1) * outer + o]*u[(i+1)*outer]) / yy[i * outer];
        }
    }
}

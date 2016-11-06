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
        for(int y = 0; y < numY; y++) {

            // explicit x
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

__device__ inline void tridagDevice1(
    REAL*   a,   // size [n]
    REAL*   b,   // size [n]
    REAL*   c,   // size [n]
    REAL*   r,   // size [n] u
    const int n,
    REAL*   u,   // size [n] u
    REAL*   yy,   // size [n] temporary
    const int mult,  // multiplier to index into arrays
    const int multU
) {
    int    i;
    REAL   beta;

    u[0]  = r[0];
    yy[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i * mult] / yy[(i-1) * mult];

        yy[i * mult] = b[i * mult] - beta*c[(i-1) * mult];
        u[i * multU]  = r[i*multU] - beta*u[(i-1) * multU];
    }

    // X) this is a backward recurrence
    u[(n-1)*multU] = u[(n-1)*multU] / yy[(n-1) * mult];
    for(i=n-2; i>=0; i--) {
        u[i*multU] = (u[i*multU] - c[i * mult]*u[(i+1)*multU]) / yy[i * mult];
    }
}

__global__ void tridag1(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarX, REAL *myDxx,
    REAL *u, REAL *yy
    ) {
    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && y < numY) {
        for(int x = 0; x < numX; x++) {
            // here a, b,c should have size [numX]
            a[o + y * outer + x * numZ * outer] =       - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 0]);
            b[o + y * outer + x * numZ * outer] = dtInv - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 1]);
            c[o + y * outer + x * numZ * outer] =       - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 2]);
        }

        // here yy should have size [numX]
        tridagDevice1(
            a + (o + y * outer),
            b + (o + y * outer),
            c + (o + y * outer),
            u + (o + y * numX * outer),
            numX,
            u + (o + y * numX * outer),
            yy+ (o + y * outer),
            numZ * outer,
            outer
        );
    }
}

__device__ inline void tridagDevice2(
    REAL*   a,   // size [n]
    REAL*   b,   // size [n]
    REAL*   c,   // size [n]
    REAL*   r,   // size [n] _y
    const int n,
    REAL*   u,   // size [n] myResult
    REAL*   yy,   // size [n] temporary
    const int mult // multiplier to index into arrays
) {
    int    i;
    REAL   beta;

    u[0]  = r[0];
    yy[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i * mult] / yy[(i-1) * mult];

        yy[i * mult] = b[i * mult] - beta*c[(i-1) * mult];
        u[i*mult]  = r[i*mult] - beta*u[(i-1)*mult];
    }

    // X) this is a backward recurrence
    u[(n-1)*mult] = u[(n-1)*mult] / yy[(n-1) * mult];
    for(i=n-2; i>=0; i--) {
        u[i*mult] = (u[i*mult] - c[i * mult]*u[(i+1)*mult]) / yy[i * mult];
    }
}

__global__ void tridag2(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarY, REAL *myDyy,
    REAL *u, REAL *v, REAL *yy, REAL *_y, REAL *myResult
    ) {

    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int x = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && x < numX) {

        REAL* u_old = u;
        u = myResult + (o + x * numY * outer);
        yy = yy + o + x * numZ * outer;

        REAL   beta;

        u[0]  = dtInv*u[(x*outer) + o] - 0.5*v[x * numY * outer + o];
        yy[0] = b[0];

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

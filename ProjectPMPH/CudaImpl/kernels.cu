__global__ void initUAndV2Dim (
                    REAL* u,
                    REAL* v,
                    REAL* myVarX,
                    REAL* myVarY,
                    REAL* myDxx,
                    REAL* myDyy,
                    REAL* myResult,
                    unsigned outer,
                    unsigned numX,
                    unsigned numY,
                    REAL dtInv
) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;

    if (o < outer && x < numX) {
        for(unsigned y = 0; y < numY; y++) {

            // explicit x
            REAL u_new = dtInv * myResult[(o*numX*numY) + (x*numY) + y];

            if (x > 0) {
                u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 0])
                            * myResult[(o*numX*numY) + ((x-1)*numY) + y];
            }
            u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 1])
                        * myResult[(o*numX*numY) + (x*numY) + y];
            if (x < numX - 1) {
                u_new += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 2])
                            * myResult[(o*numX*numY) + ((x+1)*numY) + y];
            }

            // explicit y
            REAL v_new = 0.0;
            if(y > 0) {
                v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 0])
                    *  myResult[(o*numX*numY) + (x*numY) + y-1];
            }
            v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 1])
                *  myResult[(o*numX*numY) + (x*numY) + y];
            if(y < numY - 1) {
                v_new += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 2])
                    *  myResult[(o*numX*numY) + (x*numY) + y+1];
            }
            v[(x*numY*outer) + (y*outer) + o] = v_new;
            u[(y*numX*outer) + (x*outer) + o] = u_new + v_new;
        }
    }
}

__global__ void myResultKernel2D(unsigned int outer, unsigned int numX, unsigned int numY, REAL *myX, REAL *myResult) {
	int o = threadIdx.x + blockDim.x*blockIdx.x;
  	int x = threadIdx.y + blockDim.y*blockIdx.y;

  	if (o < outer && x < numX) {
  		REAL v = max(myX[x]-(0.001*o), (REAL)0.0);
        for(unsigned y = 0; y < numY; y++) {
            myResult[o * numX * numY + x * numY + y] = v;
        }
	}
}

__global__ void myVarXYKernel(
	unsigned int numX, unsigned int numY,
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
	unsigned int outer, unsigned int numX, unsigned int numY,
	unsigned int myXindex, unsigned int myYindex,
	REAL *res, REAL *myResult
	) {
	const unsigned int o = threadIdx.x + blockDim.x * blockIdx.x;

	if (o < outer) {
        res[o] = myResult[o * numX * numY + myXindex * numY + myYindex];
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
    unsigned int outer, unsigned int numX, unsigned int numY, unsigned int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarX, REAL *myDxx,
    REAL *u, REAL *yy
    ) {
    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && y < numY) {
        for(unsigned x = 0; x < numX; x++) {
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
        u[i]  = r[i] - beta*u[(i-1)];
    }

    // X) this is a backward recurrence
    u[(n-1)] = u[(n-1)] / yy[(n-1) * mult];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i * mult]*u[(i+1)]) / yy[i * mult];
    }
}

__global__ void tridag2(
    unsigned int outer, unsigned int numX, unsigned int numY, unsigned int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarY, REAL *myDyy,
    REAL *u, REAL *v, REAL *yy, REAL *_y, REAL *myResult
    ) {

    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int x = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && x < numX) {
        int ox_idx_zo = o + x * numZ * outer;
        for(unsigned y = 0; y < numY; y++) {
            // here a, b, c should have size [numY]
            a[ox_idx_zo + y * outer] =       - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 0]);
            b[ox_idx_zo + y * outer] = dtInv - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 1]);
            c[ox_idx_zo + y * outer] =       - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 2]);
        }

        for(unsigned y = 0; y < numY; y++) {
            _y[o * numZ * numZ + x * numZ + y] = dtInv*u[(y*numX*outer) + (x*outer) + o] - 0.5*v[x * numY * outer + y * outer + o];
        }

        // here yy should have size [numY]
        tridagDevice2(
            a + (ox_idx_zo),
            b + (ox_idx_zo),
            c + (ox_idx_zo),
            _y + (o * numZ * numZ + x * numZ),
            numY,
            myResult + (o * numX * numY + x * numY),
            yy + (ox_idx_zo),
            outer
        );
    }
}

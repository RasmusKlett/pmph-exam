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
            unsigned uIdx_oyx = (o*numY * numX) + (y*numX) + x;

            u[uIdx_oyx] = dtInv * myResult[(o*numX*numY) + (x*numY) + y];

            if (x > 0) {
                u[uIdx_oyx] += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 0])
                            * myResult[(o*numX*numY) + ((x-1)*numY) + y];
            }
            u[uIdx_oyx] += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 1])
                        * myResult[(o*numX*numY) + (x*numY) + y];
            if (x < numX - 1) {
                u[uIdx_oyx] += 0.5*( 0.5*myVarX[(x*numY) + y] * myDxx[(x*4) + 2])
                            * myResult[(o*numX*numY) + ((x+1)*numY) + y];
            }

            // explicit y
            unsigned vIdx_oxy = (o*numX*numY) + (x*numY) + y;
            v[vIdx_oxy] = 0.0;
            if(y > 0) {
                v[vIdx_oxy] += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 0])
                    *  myResult[(o*numX*numY) + (x*numY) + y-1];
            }
            v[vIdx_oxy] += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 1])
                *  myResult[(o*numX*numY) + (x*numY) + y];
            if(y < numY - 1) {
                v[vIdx_oxy] += ( 0.5*myVarY[(x*numY) + y] * myDyy[(y*4) + 2])
                    *  myResult[(o*numX*numY) + (x*numY) + y+1];
            }
            u[uIdx_oyx] += v[vIdx_oxy];
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

__device__ void tridagDevice(
    REAL*   a,   // size [n]
    REAL*   b,   // size [n]
    REAL*   c,   // size [n]
    REAL*   r,   // size [n]
    const int n,
    REAL*   u,   // size [n]
    REAL*   uu   // size [n] temporary
) {
    int    i;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
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
            a[o * numZ * numZ + y * numZ + x] =       - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 0]);
            b[o * numZ * numZ + y * numZ + x] = dtInv - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 1]);
            c[o * numZ * numZ + y * numZ + x] =       - 0.5*(0.5*myVarX[x * numY + y]*myDxx[x * 4 + 2]);
        }

        // here yy should have size [numX]
        tridagDevice(
            a + (o * numZ * numZ + y * numZ),
            b + (o * numZ * numZ + y * numZ),
            c + (o * numZ * numZ + y * numZ),
            u + (o * numX * numY + y * numX),
            numX,
            u + (o * numX * numY + y * numX),
            yy+ (o * numZ * numZ + y * numZ));
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
        int ox_idx_zz = o * numZ * numZ + x * numZ;
        for(unsigned y = 0; y < numY; y++) {
            // here a, b, c should have size [numY]
            a[ox_idx_zz+y] =       - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 0]);
            b[ox_idx_zz+y] = dtInv - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 1]);
            c[ox_idx_zz+y] =       - 0.5*(0.5*myVarY[x * numY + y]*myDyy[y * 4 + 2]);
        }

        for(unsigned y = 0; y < numY; y++) {
            _y[ox_idx_zz+y] = dtInv*u[o * numY * numX + y * numX + x] - 0.5*v[o * numX * numY + x * numY + y];
        }

        // here yy should have size [numY]
        tridagDevice(
            a + (ox_idx_zz),
            b + (ox_idx_zz),
            c + (ox_idx_zz),
            _y+ (ox_idx_zz),
            numY,
            myResult + (o * numX * numY + x * numY),
            yy+ (ox_idx_zz));
    }
}
__global__ void initUAndV2Dim () {
        printf("\n###############printing from kernel\n");

}

__global__ void myResultKernel2D(unsigned int outer, unsigned int numX, unsigned int numY, REAL * myX, REAL * myResult) {
	int o = threadIdx.x + blockDim.x*blockIdx.x;
  	int x = threadIdx.y + blockDim.y*blockIdx.y;

  	if (o < outer && x < numX) {
        for(unsigned y = 0; y < numY; y++) {
            myResult[o * numX * numY + x * numY + y] = max(myX[x]-(0.001*o), (REAL)0.0);
        }
	}
}

__global__ void myVarXYKernel(
	unsigned int g, unsigned int numX, unsigned int numY,
	REAL beta, REAL nu, REAL alpha,
	REAL * myX, REAL *myY, REAL *myTimeline,
	REAL *myVarX, REAL *myVarY
	) {
	int x = threadIdx.x + blockDim.x*blockIdx.x;
  	int y = threadIdx.y + blockDim.y*blockIdx.y;

  	if (x < numX && y < numY) {
        myVarX[x * numY + y] = exp(2.0*(  beta*log(myX[x])
	                                      + myY[y]
	                                      - 0.5*nu*nu*myTimeline[g] )
	                                );
        myVarY[x * numY + y] = exp(2.0*(  alpha*log(myX[x])
	                                      + myY[y]
	                                      - 0.5*nu*nu*myTimeline[g] )
	                                ); // nu*nu
	}
}

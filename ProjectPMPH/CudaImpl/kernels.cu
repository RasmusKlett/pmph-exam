__global__ void initUAndV2Dim () {
        printf("\n###############printing from kernel\n");

}

__global__ void myResultKernel2D(unsigned int outer, unsigned int numX, unsigned int numY, REAL * myX, REAL * myResult) {
	int o = threadIdx.x + blockDim.x*blockIdx.x;
  	int x = threadIdx.y + blockDim.y*blockIdx.y;

  	if (o < outer && x < numX) {
        for(unsigned y = 0; y < numY; y++) {
            myResult[o * numX * numY + x * numY + y] = max(myX[x]-(0.001*o), (REAL)0.0);;
        }
	}
}

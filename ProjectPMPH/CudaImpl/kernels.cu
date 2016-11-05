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

__global__ void myResultKernel2D(unsigned int outer, unsigned int numX, unsigned int numY, REAL * myX, REAL * myResult) {
	int o = threadIdx.x + blockDim.x*blockIdx.x;
  	int x = threadIdx.y + blockDim.y*blockIdx.y;

  	if (o < outer && x < numX) {
        for(unsigned y = 0; y < numY; y++) {
            myResult[o * numX * numY + x * numY + y] = max(myX[x]-(0.001*o), (REAL)0.0);;
        }
	}
}

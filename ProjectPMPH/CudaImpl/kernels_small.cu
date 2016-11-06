// These kernels calculate a, b, c and _y in 3d to optimize for the small
// dataset without switching to the parallel TriDag algorithm.
// They do not seem to beat the 2d version with a and b privatized though.


__global__ void initABC3D(
        REAL* a,
        REAL* b,
        REAL* c,
        const int outer,
        const int numX,
        const int numY,
        const int numZ,
        const REAL dtInv,
        REAL* myVar,
        REAL* myD,
        const bool is_x
) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;
    int y = blockIdx.z*blockDim.z + threadIdx.z;
    if (o < outer && x < numX && y < numY) {
        const int idx = is_x ? x : y;
        a[x * numZ * outer + y * outer + o] =       - 0.5*(0.5*myVar[x * numY + y]*myD[idx * 4 + 0]);
        b[x * numZ * outer + y * outer + o] = dtInv - 0.5*(0.5*myVar[x * numY + y]*myD[idx * 4 + 1]);
        c[x * numZ * outer + y * outer + o] =       - 0.5*(0.5*myVar[x * numY + y]*myD[idx * 4 + 2]);
    }
}

__global__ void init_y(
        const int outer, 
        const int numX,
        const int numY,
        const int numZ,
        REAL* _y,
        REAL* u,
        REAL* v,
        REAL dtInv
) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    int x = blockIdx.y*blockDim.y + threadIdx.y;
    int y = blockIdx.z*blockDim.z + threadIdx.z;
    if (o < outer && x < numX && y < numY) {
        _y[(x*numZ*outer) + (y*outer) + o] = dtInv*u[(y*numX*outer) + (x*outer) + o] - 0.5*v[x * numY * outer + y * outer + o];
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

__global__ void onlyTridag1(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarX, REAL *myDxx,
    REAL *u, REAL *yy
    ) {
    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && y < numY) {

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

__global__ void onlyTridag2(
    const int outer, const int numX, const int numY, const int numZ,
    REAL *a, REAL *b, REAL *c,
    REAL dtInv,
    REAL *myVarY, REAL *myDyy,
    REAL *u, REAL *v, REAL *yy, REAL *_y, REAL *myResult
    ) {

    int o = threadIdx.x + blockDim.x*blockIdx.x;
    int x = threadIdx.y + blockDim.y*blockIdx.y;

    if (o < outer && x < numX) {
        int ox_idx_zo = o + x * numZ * outer;

        // here yy should have size [numY]
        tridagDevice2(
            a + (ox_idx_zo),
            b + (ox_idx_zo),
            c + (ox_idx_zo),
            _y + (o + x * numZ * outer),
            numY,
            myResult + (o + x * numY * outer),
            yy + (ox_idx_zo),
            outer
        );
    }
}


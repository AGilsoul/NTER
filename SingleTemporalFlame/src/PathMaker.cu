//
// Created by agils on 7/31/2023.
//

#include "../include/PathMaker.cuh"

struct Particle {
    __host__ Particle() {}
    __host__ Particle(double* X, double* U, double* grad, double val) {
        x = X[0]; y = X[1]; z = X[2];
        u = U[0]; v = U[1]; w = U[2];
        this->val = val;
        computeHostNormal(grad);
    }
    __host__ void computeHostNormal(double* grad);
    __device__ void computeNormal(double* grad);
    double x, y, z;
    double u, v, w;
    double val;
    double nx, ny, nz;
};
// basic triangle, contains pointers to surface particles that act as vertices
struct Triangle {
    __device__ void computeAvg(Particle* points);
    int v0, v1, v2;  // indices of particle vertices
    double avgX, avgY, avgZ;
    double avgU, avgV, avgW;
    double avgNX, avgNY, avgNZ;
    double avgVal;
};

__host__ double computeHostMagnitude(double* arr) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += pow(arr[i], 2);
    }
    return pow(sum, 0.5);
}

__device__ double computeMagnitude(double* arr) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += pow(arr[i], 2);
    }
    return pow(sum, 0.5);
}

__host__ void calcHostNormal(double* normal, double* gradient) {
    double mag = computeHostMagnitude(gradient);
    for (int i = 0; i < 3; i++) {
        normal[i] = gradient[i] / mag;
    }
}

__device__ void calcNormal(double* normal, double* gradient) {
    double mag = computeMagnitude(gradient);
    for (int i = 0; i < 3; i++) {
        normal[i] = gradient[i] / mag;
    }
}

__host__ double arrMin(double* arr, int len) {
    double curMin = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] < curMin) curMin = arr[i];
    }
    return curMin;
}

__host__ double arrMax(double* arr, int len) {
    double curMax = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > curMax) curMax = arr[i];
    }
    return curMax;
}

__host__ void Particle::computeHostNormal(double* grad)
{
    double res[3];
    calcHostNormal(res, grad);
    nx = res[0];
    ny = res[1];
    nz = res[2];
}

__device__ void Particle::computeNormal(double* grad)
{
    double res[3];
    calcNormal(res, grad);
    nx = res[0];
    ny = res[1];
    nz = res[2];
}

__device__ void Triangle::computeAvg(Particle *points) {
    auto p0 = points[v0]; auto p1 = points[v1]; auto p2 = points[v2];
    avgX = (p0.x + p1.x + p2.x) / 3; avgY = (p0.y + p1.y + p2.y) / 3; avgZ = (p0.z + p1.z + p2.z) / 3;
    avgU = (p0.u + p1.u + p2.u) / 3; avgV = (p0.v + p1.v + p2.v) / 3; avgW = (p0.w + p1.w + p2.w) / 3;
    avgNX = (p0.nx + p1.nx + p2.nx) / 3; avgNY = (p0.ny + p1.ny + p2.ny) / 3; avgNZ = (p0.nz + p1.nz + p2.nz) / 3;
    avgVal = (p0.val + p1.val + p2.val) / 3;
}

__device__ double dotProduct(double* v1, double* v2) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

__device__ void crossProduct(double* res, double* v1, double* v2) {
    res[0] = v1[1]*v2[2] - v1[2]*v2[1];
    res[1] = v1[2]*v1[0] - v1[0]*v2[2];
    res[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// for every combination of simplices on the two surfaces
// results stored in 2d array with numStartSimplices rows and numTargetSimplices columns
__global__ void triangleIntersects(int* interIndices, double* interDists, Particle particle, Triangle* targetTriangles, Particle* targetParticles, int numTargetTris) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTargetTris) return;
    double threshold = 0.0001;
    Triangle targSimplex = targetTriangles[index];
    Particle v0Targ = targetParticles[targSimplex.v0]; Particle v1Targ = targetParticles[targSimplex.v1]; Particle v2Targ = targetParticles[targSimplex.v2];
    if (index == 97971) {
        printf("triangle %d v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", index, targSimplex.v0, v0Targ.x, v0Targ.y, v0Targ.z, v0Targ.val);
    }
    // get normal vector of simplex by taking average normal of vertices
    double avgNormal[3] = {particle.nx + particle.u, particle.ny + particle.v, particle.nz + particle.w};
    // get center of simplex by taking average position of vertices
    double avgX[3] = {particle.x, particle.y, particle.z};
    double avgTargX[3] = {targSimplex.avgX, targSimplex.avgY, targSimplex.avgZ};

    if (targSimplex.avgVal == 0) {
        //printf("target simplex %d avgVal: %f, vertices: (%d,%d,%d)\n", index, targSimplex.avgVal, targSimplex.v0, targSimplex.v1, targSimplex.v2);
    }
    double A[3] = {v0Targ.x, v0Targ.y, v0Targ.z};
    double AC[3] = {v1Targ.x - v0Targ.x, v1Targ.y - v0Targ.y, v1Targ.z - v0Targ.z};
    double AB[3] = {v2Targ.x - v0Targ.x, v2Targ.y - v0Targ.y, v2Targ.z - v0Targ.z};
    double BC[3] = {v1Targ.x - v2Targ.x, v1Targ.y - v2Targ.y, v1Targ.z - v2Targ.z};
    //printf("A: (%f,%f,%f)\n", A[0], A[1], A[2]);
    //printf("AB: (%f,%f,%f), AC: (%f,%f,%f)\n", AB[0], AB[1], AB[2], AC[0], AC[1], AC[2]);
    double planeNormal[3];
    crossProduct(planeNormal, AB, AC);
    double magnitude = computeMagnitude(planeNormal);
    //printf("magnitude of cross prod (%f,%f,%f): %f\n", planeNormal[0], planeNormal[1], planeNormal[2], magnitude);
    for (int j = 0; j < 3; j++) {
        planeNormal[j] /= magnitude;
    }
    //printf("normal to triangle %d plane: (%f,%f,%f)\n", index, planeNormal[0], planeNormal[1], planeNormal[2]);

    double angleDotProd = dotProduct(avgNormal, planeNormal);
    if (angleDotProd == 0) {
        interIndices[index] = 0;
        return;
    }

    double Q[3];
    double t = (dotProduct(planeNormal, A) - dotProduct(planeNormal, avgX)) / dotProduct(planeNormal, avgNormal);

    for (int j = 0; j < 3; j++) {
        Q[j] = avgX[j] + avgNormal[j] * t;
    }
    //printf("point Q on triangle %d plane: (%f,%f,%f)\n", index, Q[0], Q[1], Q[2]);

    double crossABQ[3], crossACQ[3], crossBCQ[3];
    crossProduct(crossABQ, AB,Q);
    crossProduct(crossACQ, AC,Q);
    crossProduct(crossBCQ, BC,Q);

    double dotAB = dotProduct(crossABQ, planeNormal);
    double dotAC = dotProduct(crossACQ, planeNormal);
    double dotBC = dotProduct(crossBCQ, planeNormal);

    //printf("index %d, dotAB=%f, dotAC=%f, dotBC=%f\n", index, dotAB, dotAC, dotBC);

    bool inAB = (dotAB > 0);
    bool inBC = (dotBC > 0);
    bool inAC = (dotAC > 0);
    double curMinAvgX[3];
    for (int j = 0; j < 3; j++)
    {
        curMinAvgX[j] = avgX[j] - avgTargX[j];
    }
    double dist = computeMagnitude(curMinAvgX);

    if (inAB && inAC && inBC && Q[0] > 0 && Q[1] > 0 && Q[2] > 0 && dist <= threshold)
    {
        // if intersection
        /*
        printf("intersection found at index %d\n", index);
        printf("index %d, dist from particle to triangle avg: %f\n", index, dist);
        printf("index %d, dotAB=%f, dotAC=%f, dotBC=%f\n", index, dotAB, dotAC, dotBC);
        printf("index %d, A of intersection triangle: (%f,%f,%f)\n", index,
               A[0],
               A[1],
               A[2]);
        printf("index %d, plane normal of intersection triangle: (%f,%f,%f)\n", index, planeNormal[0], planeNormal[1], planeNormal[2]);
        printf("index %d,intersection Q with t=%f: (%f,%f,%f)\n", index, t, Q[0], Q[1], Q[2]);
        */
        interIndices[index] = 1;
        interDists[index] = dist;
        return;
    }
    interIndices[index] = 0;
}

double getClosestDelta(int* interIndices, double* interDists, Particle particle, Triangle* triangles, int numTriangles) {
    //printf("in getClosestDelta ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    // find closest intersect
    int interIndex = -1;
    double minDist = 0;

    for (int i = 0; i < numTriangles; i++) {
        //printf("in loop at triangle index %d, error code %d\n", i, cudaDeviceSynchronize());
        double curDist = interDists[i];
        if (interIndices[i] == 1) {
            //printf("found intersection with simplex %d at distance %f\n", i, curDist);
            if (interIndex == -1) {
                minDist = curDist;
                interIndex = i;
            }
            else if (curDist < minDist) {
                minDist = curDist;
                interIndex = i;
            }
        }
    }

    //printf("done getting closest intersect\n");
    if (interIndex == -1) {
        //printf("No intersection, point goes out of range!\n");
        return -100000;
    }
    //printf("calculating delta\n");
    Triangle interTriangle = triangles[interIndex];
    //printf("found closest triangle with point indices (%d,%d,%d) and avg val=%f\n", interTriangle.v0, interTriangle.v1, interTriangle.v2, interTriangle.avgVal);
    return interTriangle.avgVal - particle.val;
    //printf("Got delta of triangle %d: %f - %f = %f\n", index, interTriangle.avgVal, curTriangle.avgVal, deltas[index]);
}

// assign particles to surfaces, assign values
__global__ void initParticles(Particle* particles, double* X, double* val, double* grad, int numParticles, bool hasGrad) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    int arrIndex = index * 3;
    particles[index].x = X[arrIndex];
    particles[index].y = X[arrIndex+1];
    particles[index].z = X[arrIndex+2];
    particles[index].val = val[arrIndex];
    if (index == 21863) {
        printf("v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", index, X[arrIndex], X[arrIndex+1], X[arrIndex+2], val[index]);
    }

    if (hasGrad) {
        double curGrad[3] = {grad[arrIndex], grad[arrIndex+1], grad[arrIndex+2]};
        particles[index].computeNormal(curGrad);
    }
}

// assign simplices to surfaces, assign values
__global__ void initTriangles(Triangle* triangles, Particle* particles, int* triIndices, int numTriangles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTriangles) return;
    int arrIndex = index * 3;
    int curIndices[3] = {triIndices[arrIndex], triIndices[arrIndex+1], triIndices[arrIndex+2]};
    triangles[index].v0 = curIndices[0];
    triangles[index].v1 = curIndices[1];
    triangles[index].v2 = curIndices[2];
    triangles[index].computeAvg(particles);
}

__global__ void getAvgs(double* avgs, Triangle* triangles, int numTriangles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numTriangles) return;
    avgs[index] = triangles[index].avgVal;
}

// do this
void fillSurface(Triangle* triangles, int* triIndices, Particle* particles, int numParticles, int numTriangles, double* X, double* val, double* grad = nullptr) {
    int blockSize = 256;
    int numParticleBlocks = (numParticles + blockSize - 1) / blockSize;
    int numTriangleBlocks = (numTriangles + blockSize - 1) / blockSize;
    //printf("in fill surface function\n");
    //printf("\ttriangle 0 vertices: (%d,%d,%d)\n", triangles[0].v0, triangles[0].v1, triangles[0].v2);
    size_t partArrSize = sizeof(Particle) * numParticles;
    size_t triArrSize = sizeof(Triangle) * numTriangles;
    size_t dubParticleArrSize = sizeof(double) * numParticles;
    size_t intTriArrSize = sizeof(int) * numTriangles;
    bool hasGrad = false;
    Particle* gpuParticles;
    Triangle* gpuTriangles;
    double *gpuX, *gpuVal, *gpuGrad;
    int *gpuTriIndices;
    cudaMalloc((void**)&gpuParticles, partArrSize);
    printf("cudaMalloc 1: error code %d\n", cudaDeviceSynchronize());
    cudaMalloc((void**)&gpuX, dubParticleArrSize * 3);
    printf("cudaMalloc 2: error code %d\n", cudaDeviceSynchronize());
    cudaMalloc((void**)&gpuVal, dubParticleArrSize);
    printf("cudaMalloc 3: error code %d\n", cudaDeviceSynchronize());

    if (grad != nullptr) {
        cudaMalloc((void**)&gpuGrad, dubParticleArrSize * 3);
        hasGrad = true;
    }
    else {
        gpuGrad = nullptr;
    }
    printf("fill surface pre copy v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", 21863, X[21863 * 3], X[21863 * 3 + 1], X[21863 * 3 + 2], val[21863]);

    cudaMemcpy(gpuParticles, particles, partArrSize, cudaMemcpyHostToDevice);
    printf("copied particles: error code %d\n", cudaDeviceSynchronize());
    cudaMemcpy(gpuX, X, dubParticleArrSize * 3, cudaMemcpyHostToDevice);
    printf("copied X: error code %d\n", cudaDeviceSynchronize());
    cudaMemcpy(gpuVal, val, dubParticleArrSize, cudaMemcpyHostToDevice);
    if (grad != nullptr) cudaMemcpy(gpuGrad, grad, dubParticleArrSize * 3, cudaMemcpyHostToDevice);
    initParticles<<<numParticleBlocks, blockSize>>>(gpuParticles, gpuX, gpuVal, gpuGrad, numParticles, hasGrad);
    cudaDeviceSynchronize();
    cudaFree(gpuX); cudaFree(gpuVal); cudaFree(gpuGrad);
    cudaMalloc((void**)&gpuTriangles, triArrSize);
    cudaMalloc((void**)&gpuTriIndices, intTriArrSize * 3);
    cudaMemcpy(gpuTriangles, triangles, triArrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTriIndices, triIndices, intTriArrSize * 3, cudaMemcpyHostToDevice);
    initTriangles<<<numTriangleBlocks, blockSize>>>(gpuTriangles, gpuParticles, gpuTriIndices, numTriangles);
    cudaDeviceSynchronize();
    cudaMemcpy(triangles, gpuTriangles, triArrSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles, gpuParticles, partArrSize, cudaMemcpyDeviceToHost);
    Particle testParticle = particles[21863];
    printf("fill surface post copy v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", 21863, testParticle.x, testParticle.y, testParticle.z, testParticle.val);

    //printf("done copying triangles!\n");
    //printf("\ttriangle index 0 vertices: (%d,%d,%d)\n", triangles[0].v0, triangles[0].v1, triangles[0].v2);
    cudaDeviceSynchronize();
    cudaFree(gpuTriangles); cudaFree(gpuTriIndices); cudaFree(gpuParticles);
}

double calcDelta(Particle particle, Particle* targParticles, Triangle* targTriangles, int numTriangles, int numParticles) {
    int blockSize = 256;
    int numBlocks = (numTriangles + blockSize - 1) / blockSize;
    //printf("calculating delta:\n");
    //printf("\tX at p0: (%f,%f,%f)\n", particle.x, particle.y, particle.z);
    //printf("\tnormal at p0: (%f,%f,%f)\n", particle.nx, particle.ny, particle.nz);
    Particle *gpuTargParticles;
    Triangle *gpuTargTriangles;
    int* interIndices = (int*)malloc(sizeof(int) * numTriangles);
    int* gpuInterIndices;
    double* interDists = (double*)malloc(sizeof(double) * numTriangles);
    double* gpuInterDists;
    //printf("cuda malloc ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    cudaMalloc((void**)&gpuTargParticles, sizeof(Particle) * numParticles);
    cudaMalloc((void**)&gpuTargTriangles, sizeof(Triangle) * numTriangles);
    cudaMalloc((void**)&gpuInterIndices, sizeof(int) * numTriangles);
    cudaMalloc((void**)&gpuInterDists, sizeof(double) * numTriangles);

    //printf("cuda memcpy ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    cudaMemcpy(gpuTargParticles, targParticles, sizeof(Particle) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTargTriangles, targTriangles, sizeof(Triangle) * numTriangles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInterIndices, interIndices, sizeof(int) * numTriangles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInterDists, interDists, sizeof(double) * numTriangles, cudaMemcpyHostToDevice);

    //printf("triangle intersects kernel ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    Particle testParticle = targParticles[21863];
    printf("v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", 21863, testParticle.x, testParticle.y, testParticle.z, testParticle.val);

    triangleIntersects<<<numBlocks, blockSize>>>(gpuInterIndices, gpuInterDists, particle, gpuTargTriangles, gpuTargParticles, numTriangles);
    //printf("cuda memcpy to host ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    cudaMemcpy(interIndices, gpuInterIndices, sizeof(int) * numTriangles, cudaMemcpyDeviceToHost);
    cudaMemcpy(interDists, gpuInterDists, sizeof(double) * numTriangles, cudaMemcpyDeviceToHost);

    //printf("cuda free ...\n");
    //printf("error code %d\n", cudaDeviceSynchronize());
    cudaFree(gpuTargParticles); cudaFree(gpuTargTriangles);
    cudaFree(gpuInterDists); cudaFree(gpuInterIndices);

    return getClosestDelta(interIndices, interDists, particle, targTriangles, numTriangles);

}


DELLEXPORT double getDelta(double* xyz, double* uvw, double* gxgygz, double val, int* targTri, double* X, double* vals, int numTriangles, int numParticles, double dt) {
    //printf("entered func: X=(%f,%f,%f)\n", xyz[0], xyz[1], xyz[2]);
    //printf("\ttriangle 0 indices: (%d,%d,%d)\n", targTri[0], targTri[1], targTri[2]);
    Particle particle(xyz, uvw, gxgygz, val);
    //printf("entered func:\n");
    printf("\tX at p0: (%f,%f,%f)\n", particle.x, particle.y, particle.z);
    //printf("\tnormal at p0: (%f,%f,%f)\n", particle.nx, particle.ny, particle.nz);
    Particle* cpuParticles = (Particle*)malloc(sizeof(Particle) * numParticles);
    Triangle* cpuTriangles = (Triangle*)malloc(sizeof(Triangle) * numTriangles);
    //cout << "Filling surface with " << numTriangles << " triangles and " << numParticles << " points, error code " << cudaGetLastError() << " ..." << endl;
    fillSurface(cpuTriangles, targTri, cpuParticles, numParticles, numTriangles, X, vals);
    Particle testParticle = cpuParticles[21863];
    printf("dellexport v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", 21863, testParticle.x, testParticle.y, testParticle.z, testParticle.val);

    //printf("Triangle 0 vertex indices: (%d,%d,%d)\n", cpuTriangles[0].v0, cpuTriangles[0].v1, cpuTriangles[0].v2);
    //cout << "Calculating deltas, error code " << cudaGetLastError() << " ..." << endl;

    double delRes = calcDelta(particle, cpuParticles, cpuTriangles, numTriangles, numParticles);
    //cout << "delta: " << delRes << ", error code "  << cudaGetLastError() <<endl;
    return delRes;
}

DELLEXPORT double* getAvgTriVals(int* tri, double* X, double* U, double* val, int numTriangles, int numParticles) {
    Triangle* cpuTriangles = (Triangle*)malloc(sizeof(Triangle) * numTriangles);
    Particle* cpuParticles = (Particle*)malloc(sizeof(Particle) * numParticles);
    fillSurface(cpuTriangles, tri, cpuParticles, numParticles, numTriangles, X, U, val);
    double* avgVals = (double*)malloc(sizeof(double) * numTriangles);
    double* gpuAvgs;
    cudaMalloc((void**)&gpuAvgs, sizeof(double) * numTriangles);
    Triangle* gpuTriangles;
    cudaMalloc((void**)&gpuTriangles, sizeof(Triangle) * numTriangles);
    cudaMemcpy(gpuTriangles, cpuTriangles, sizeof(Triangle) * numTriangles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuAvgs, avgVals, sizeof(double) * numTriangles, cudaMemcpyHostToDevice);
    getAvgs<<<256, (numTriangles + 256 - 1) / 256>>>(gpuAvgs, gpuTriangles, numTriangles);
    cudaDeviceSynchronize();
    cudaMemcpy(avgVals, gpuAvgs, sizeof(double) * numTriangles, cudaMemcpyDeviceToHost);
    return avgVals;
}



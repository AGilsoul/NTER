//
// Created by agils on 8/8/2023.
//

#include "../include/PathMaker.h"

double computeMagnitude(vector<double>& arr) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += pow(arr[i], 2);
    }
    return pow(sum, 0.5);
}

double computeMagnitude(double* arr) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += pow(arr[i], 2);
    }
    return pow(sum, 0.5);
}

double dotProduct(vector<double>& v1, vector<double>& v2) {
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

vector<double> crossProduct(vector<double>& v1, vector<double>& v2) {
    vector<double> res(3);
    res[0] = v1[1]*v2[2] - v1[2]*v2[1];
    res[1] = v1[2]*v1[0] - v1[0]*v2[2];
    res[2] = v1[0]*v2[1] - v1[1]*v2[0];
    return res;
}

struct Particle {
    Particle() {
        X = vector<double>(3);
        U = vector<double>(3);
        n = vector<double>(3);
    }
    Particle(vector<double>& X, vector<double>& U, vector<double>& grad, double K, double T, bool normCalculated) {
        this->X = vector<double>(3);
        this->U = vector<double>(3);
        this->n = vector<double>(3);
        for (int i = 0; i < 3; i++) {
            this->X[i] = X[i];
            this->U[i] = U[i];
        }
        this->K = K;
        this->T = T;
        this->r = 1/K;
        computeNormal(grad, normCalculated);
    }
    void computeNormal(vector<double>& grad, bool normCalculated) {
        if (normCalculated) {
            for (int i = 0; i < 3; i++) {
                n[i] = grad[i];
            }
        }
        double mag = computeMagnitude(grad);
        for (int i = 0; i < 3; i++) {
            n[i] = grad[i] / mag;
        }
    }
    vector<double> X;
    vector<double> U;
    double K, T, r;
    vector<double> n;
};

struct Triangle {
    Triangle() {}
    void computeAvg(vector<Particle>& points) {
        auto p0 = points[v0]; auto p1 = points[v1]; auto p2 = points[v2];
        for (int i = 0; i < 3; i++){
            avgX[i] = (p0.X[i] + p1.X[i] + p2.X[i]) / 3;
            avgU[i] = (p0.U[i] + p1.U[i] + p2.U[i]) / 3;
            avgN[i] = (p0.n[i] + p1.n[i] + p2.n[i]) / 3;
        }
        double avgNMag = computeMagnitude(avgN);
        for (int i = 0; i < 3; i++) {
            avgN[i] /= avgNMag;
        }
        avgK = (p0.K + p1.K + p2.K) / 3; avgT = (p0.T + p1.T + p2.T) / 3; avgr = (p0.r + p1.r + p2.r) / 3;
    }
    int v0, v1, v2;
    double avgX[3];
    double avgU[3];
    double avgN[3];
    double avgK, avgT, avgr;

};

vector<double> calcDelta(Particle p0, vector<Particle>& surfaceParticles, vector<Triangle>& surfaceTriangles, int numTriangles, int numParticles, double dt) {
    int interIndex = -1;
    double minDist = 0;
    double threshold = 0.0001;
    for (int i = 0; i < numTriangles; i++) {
        Triangle targTriangle = surfaceTriangles[i];
        Particle v0 = surfaceParticles[targTriangle.v0];
        Particle v1 = surfaceParticles[targTriangle.v1];
        Particle v2 = surfaceParticles[targTriangle.v2];
        if (abs(targTriangle.avgX[2] - v0.X[2]) > threshold) continue;

        bool compute = true;
        for (int j = 0; j < 3; j++) {
            if (abs(p0.X[j] - targTriangle.avgX[j]) > threshold) {
                compute = false;
            }
        }
        if (!compute) continue;


        vector<double> A = {v0.X[0], v0.X[1], v0.X[2]};
        vector<double> AC = {v1.X[0] - v0.X[0], v1.X[1] - v0.X[1], v1.X[2] - v0.X[2]};
        vector<double> AB = {v2.X[0] - v0.X[0], v2.X[1] - v0.X[1], v2.X[2] - v0.X[2]};
        vector<double> BC = {v1.X[0] - v2.X[0], v1.X[1] - v2.X[1], v1.X[2] - v2.X[2]};

        vector<double> planeNormal = crossProduct(AB, AC);
        double magnitude = computeMagnitude(planeNormal);
        for (int j = 0; j < 3; j++)
        {
            planeNormal[j] /= magnitude;
        }

        vector<double> path(3);
        for (int j = 0; j < 3; j++) {
            path[j] = p0.n[j] + p0.U[j] * dt;
        }

        double angleDotProd = dotProduct(path, planeNormal);
        if (angleDotProd == 0) {
            continue;
        }

        vector<double> Q(3);
        double t = (dotProduct(planeNormal, A) - dotProduct(planeNormal, p0.X)) / dotProduct(planeNormal, path);
        for (int j = 0; j < 3; j++)
        {
            Q[j] = p0.X[j] + path[j] * t;
        }

        vector<double> crossABQ = crossProduct(AB, Q);
        vector<double> crossACQ = crossProduct(AC, Q);
        vector<double> crossBCQ = crossProduct(BC, Q);

        double dotAB = dotProduct(crossABQ, planeNormal);
        double dotAC = dotProduct(crossACQ, planeNormal);
        double dotBC = dotProduct(crossBCQ, planeNormal);

        bool inAB = (dotAB > 0);
        bool inBC = (dotBC > 0);
        bool inAC = (dotAC > 0);
        vector<double> curMinAvgX(3);
        for (int j = 0; j < 3; j++)
        {
            curMinAvgX[j] = p0.X[j] - targTriangle.avgX[j];
        }
        double dist = computeMagnitude(curMinAvgX);

        if (inAB && inAC && inBC && Q[0] > 0 && Q[1] > 0 && Q[2] > 0 && dist <= threshold)
        {
            if (interIndex == -1) {
                interIndex = i;
                minDist = dist;
            }
            else if (dist < minDist) {
                interIndex = i;
                minDist = dist;
            }
        }
    }
    vector<double> interData(13);
    if (interIndex == -1) {
        for (int i = 0; i < 13; i++) {
            interData[i] = INVALID;
        }
    }
    else {
        Triangle targTri = surfaceTriangles[interIndex];
        interData[0] = targTri.avgX[0]; interData[1] = targTri.avgX[1]; interData[2] = targTri.avgX[2];
        interData[3] = targTri.avgU[0]; interData[4] = targTri.avgU[1]; interData[5] = targTri.avgU[2];
        interData[6] = targTri.avgN[0]; interData[7] = targTri.avgN[1]; interData[8] = targTri.avgN[2];
        interData[9] = targTri.avgK; interData[10] = targTri.avgT; interData[11] = targTri.avgr;
        interData[12] = targTri.avgK - p0.K;
    }
    return interData;
}

// assign particles to surfaces, assign values
void initParticles(vector<Particle>& particles, vector<double>& X, vector<double>& U, vector<double>& K, vector<double>& T, const int numParticles) {
    for (int i = 0; i < numParticles; i++) {
        Particle curParticle;
        int arrIndex = i * 3;
        for (int j = 0; j < 3; j++) {
            curParticle.X[j] = X[arrIndex + j];
            curParticle.U[j] = U[arrIndex + j];
        }
        curParticle.K = K[i];
        curParticle.T = T[i];
        particles.push_back(curParticle);
    }
}

void initTriangles(vector<Triangle>& triangles, vector<Particle>& particles, vector<int>& triIndices, const int numTriangles) {
    for (int i = 0; i < numTriangles; i++) {
        int arrIndex = i * 3;
        Triangle curTriangle;
        curTriangle.v0 = triIndices[arrIndex];
        curTriangle.v1 = triIndices[arrIndex+1];
        curTriangle.v2 = triIndices[arrIndex+2];
        curTriangle.computeAvg(particles);
        triangles.push_back(curTriangle);
    }
}

void fillSurface(vector<Triangle>& triangles, vector<int>& triIndices, vector<Particle>& particles, int numParticles, int numTriangles, vector<double>& X, vector<double>& U, vector<double>& K, vector<double>& T) {
    //printf("fill surface p0.val=%f, p1.val=%f, p2.val=%f\n", val[25011], val[25008], val[25007]);
    //printf("in fill surface function\n");
    //printf("\ttriangle 0 vertices: (%d,%d,%d)\n", triangles[0].v0, triangles[0].v1, triangles[0].v2);
    //cout << "Filling surface particles ..." << endl;
    initParticles(particles, X, U, K, T, numParticles);
    //cout << "Filling surface triangles ..." << endl;
    initTriangles(triangles, particles, triIndices, numTriangles);
}

// returns array of size 13, including intercept xyz, uvw, nxnynz, avg K, avg T, avg r
DELLEXPORT double* getDelta(double* xyz, double* uvw, double* gxgygz, double K0, double T0, int* targTri, double* X, double* U, double* K, double* T, int numTriangles, int numParticles, double dt, bool normCalculated) {
    vector<int> locTris;
    vector<double> locXYZ, locUVW, locGXGYGZ, locX, locU, locKs, locTs;
    for (int i = 0; i < 3; i++) {
        locXYZ.push_back(xyz[i]);
        locUVW.push_back(uvw[i]);
        locGXGYGZ.push_back(gxgygz[i]);
    }
    for (int i = 0; i < numTriangles * 3; i++) locTris.push_back(targTri[i]);
    for (int i = 0; i < numParticles; i++) {
        int curArrIndex = i * 3;
        for (int j = 0; j < 3; j++) {
            locX.push_back(X[curArrIndex + j]);
            locU.push_back(U[curArrIndex + j]);
        }
        locKs.push_back(K[i]);
        locTs.push_back(T[i]);
    }
    Particle particle(locXYZ, locUVW, locGXGYGZ, K0, T0, normCalculated);
    vector<Particle> surfaceParticles;
    vector<Triangle> surfaceTriangles;
    //cout << "Filling surface with " << numTriangles << " triangles and " << numParticles << " points ..." << endl;
    fillSurface(surfaceTriangles, locTris, surfaceParticles, numParticles, numTriangles, locX, locU, locKs, locTs);
    // Particle testParticle = surfaceParticles[21863];
    // printf("dellexport v0: %d X: (%f,%f,%f), v0 avgVal: %f\n", 21863, testParticle.X[0], testParticle.X[1], testParticle.X[2], testParticle.val);

    //printf("Triangle 0 vertex indices: (%d,%d,%d)\n", cpuTriangles[0].v0, cpuTriangles[0].v1, cpuTriangles[0].v2);
    //cout << "Calculating deltas, error code " << cudaGetLastError() << " ..." << endl;

    vector<double> delRes = calcDelta(particle, surfaceParticles, surfaceTriangles, numTriangles, numParticles, dt);
    double* returnArr = (double*)malloc(sizeof(double) * 13);
    for (int i = 0; i < 13; i++) {
        returnArr[i] = delRes[i];
    }
    //cout << "delta: " << delRes << endl;
    return returnArr;
}

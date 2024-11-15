#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include <regex>
#include <cuda_runtime.h>
#include "nlohmann/json.hpp"
#include <opencv2/core.hpp>       // Core module (e.g., cv::Mat)
#include <opencv2/imgcodecs.hpp>  // Image reading/writing functions
#include <opencv2/imgproc.hpp>  
#include <memory>
# define BLOCK_SIZE 1024
# define SENSOR_WIDTH 36.0f //TODO: if in the future, meshroon changes, please also change this value, default of meshroom is 36.0f


__constant__ float d_direction[3];
__constant__ float d_up[3];
__constant__ float d_right[3];
__constant__ float d_startPoint[3];
__constant__ float d_rotation[9];
__constant__ float d_nrotation[9];
__constant__ float d_camera[2]; // focal length and sensor width in mm

bool isVertexPositionsLine(const std::string& line) {
    // Regular expression to match the format "# <number> vertex positions"
    std::regex pattern(R"(^# \d+ vertex positions$)");
    return std::regex_match(line, pattern);
}

bool isMeshFacesLine(const std::string& line) {
    // Regular expression to match the format "# Mesh '' with <number> faces"
    std::regex pattern(R"(^# Mesh '' with \d+ faces$)");
    return std::regex_match(line, pattern);
}

__device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}   

__global__ void computeMaskArea3D(float3 * vertices, uint3 * faces, float * areas, int numFaces, int * yindices, int * xindices, int numIndices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numFaces) {
        areas[idx] = 0.0f;
        float3 v0 = vertices[faces[idx].x];
        float3 v1 = vertices[faces[idx].y];
        float3 v2 = vertices[faces[idx].z]; 
        v0 = make_float3(
            d_nrotation[0] * v0.x + d_nrotation[1] * v0.y + d_nrotation[2] * v0.z,
            d_nrotation[3] * v0.x + d_nrotation[4] * v0.y + d_nrotation[5] * v0.z,
            d_nrotation[6] * v0.x + d_nrotation[7] * v0.y + d_nrotation[8] * v0.z
        );
        v1 = make_float3(
            d_nrotation[0] * v1.x + d_nrotation[1] * v1.y + d_nrotation[2] * v1.z,
            d_nrotation[3] * v1.x + d_nrotation[4] * v1.y + d_nrotation[5] * v1.z,
            d_nrotation[6] * v1.x + d_nrotation[7] * v1.y + d_nrotation[8] * v1.z
        );
        v2 = make_float3(
            d_nrotation[0] * v2.x + d_nrotation[1] * v2.y + d_nrotation[2] * v2.z,
            d_nrotation[3] * v2.x + d_nrotation[4] * v2.y + d_nrotation[5] * v2.z,
            d_nrotation[6] * v2.x + d_nrotation[7] * v2.y + d_nrotation[8] * v2.z
        );
        
        for(int i = 0; i < numIndices ; i++){
            float3 temp1 = make_float3(d_direction[0] * d_camera[0], d_direction[1] * d_camera[0], d_direction[2] * d_camera[0]);
            float dx = xindices[i] - 1344.f / 2.f;
            float dy = 1344.f / 2.f - yindices[i];
            dx *= d_camera[1] / 1344.f;
            dy *= d_camera[1] / 1344.f;
            float3 temp2 = make_float3(d_right[0] * dx, d_right[1] * dx, d_right[2] * dx);
            float3 temp3 = make_float3(d_up[0] * dy, d_up[1] * dy, d_up[2] * dy);
            float3 D = make_float3(
                temp1.x + temp2.x + temp3.x,
                temp1.y + temp2.y + temp3.y,
                temp1.z + temp2.z + temp3.z
            );
            float magnitudeD = sqrtf(D.x * D.x + D.y * D.y + D.z * D.z);
            D.x /= magnitudeD;
            D.y /= magnitudeD;
            D.z /= magnitudeD;
            float3 O = make_float3(d_startPoint[0], d_startPoint[1], d_startPoint[2]);
            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);         
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            float3 P = cross(D, e2);
            float det = dot(e1, P);
            if (det > -1e-6 && det < 1e-6) {
                continue; // this ray is parallel to this triangle
            }
            float invDet = 1.0f / det;
            float3 T = make_float3(O.x - v0.x, O.y - v0.y, O.z - v0.z);
            float u = dot(T, P) * invDet;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            float3 Q = cross(T, e1);
            float v = dot(D, Q) * invDet;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            float t = dot(e2, Q) * invDet;
            if (t > 1e-6) {
                float3 crossProduct = cross(e1, e2);
                float area = 0.5f * sqrtf(crossProduct.x * crossProduct.x + crossProduct.y * crossProduct.y + crossProduct.z * crossProduct.z);
                areas[idx] = area;
                break;
            }
        }

    }

}

__global__ void computeTriangleAreas(float3 * vertices, uint3 * faces, float * areas, int numFaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numFaces) {
        float3 v0 = vertices[faces[idx].x];
        float3 v1 = vertices[faces[idx].y];
        float3 v2 = vertices[faces[idx].z]; 
        float3 vector0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 vector1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        
        float3 crossProduct = make_float3(
            vector0.y * vector1.z - vector0.z * vector1.y,
            vector0.z * vector1.x - vector0.x * vector1.z,
            vector0.x * vector1.y - vector0.y * vector1.x
        );
        
        float area = 0.5f * sqrtf(crossProduct.x * crossProduct.x + crossProduct.y * crossProduct.y + crossProduct.z * crossProduct.z);

        areas[idx] = area;
    }
}

__global__ void reduceSum(float * areas, float * totalArea, int numFaces) {
    __shared__ float sAreas[BLOCK_SIZE];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Initialize shared memory
    sAreas[threadIdx.x] = (idx < numFaces) ? areas[idx] : 0.0f;
    __syncthreads();
    // Perform the reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sAreas[threadIdx.x] += sAreas[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(totalArea, sAreas[0]);
    }
}

void readSFMData(const std::string & path, const std::string & viewId, std::vector<float> & R, std::vector<float> & O, float & focalLength) {
    std::ifstream file(path);
    if(!file.is_open()) {
        std::cerr << "Error: could not open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    nlohmann::json jsonData;
    file >> jsonData;
    const auto & poses = jsonData["poses"];
    for (const auto& pose : poses) {
        std::string poseId = pose["poseId"];
        if (poseId == viewId) {
            const auto & rotation = pose["pose"]["transform"]["rotation"];
            for (const std::string r : rotation) {
                R.push_back(std::stof(r));
            }
            const auto & translation = pose["pose"]["transform"]["center"];
            for (const std::string o : translation) {
                O.push_back(std::stof(o));
            }
        }
    }
    std::string fl = jsonData["intrinsics"][0]["focalLength"];
    focalLength = std::stof(fl);
}


void cropMesh(const std::string & inputPath, const std::string & outputPath, const float * areas){
    std::ifstream inputFile(inputPath);
    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: could not open output file " << outputPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    int index = 0;
    while (std::getline(inputFile, line)) {
        if(line.substr(0, 3) == "f  ") {
            if(areas[index] == 0.0f) {
                index++;
                continue;
            } else {
                index++;
            }
        }
        outputFile << line << std::endl;
    }

    outputFile.close();
}


int main(int argc, char* argv[]) {
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<float[]> direction(new float[3]);
    std::unique_ptr<float[]> startPoint(new float[3]); 
    std::unique_ptr<float[]> up(new float[3]);
    std::unique_ptr<float[]> right(new float[3]); 
    cv::Mat mask;
    std::string filePath = argv[1];
    std::ifstream file(filePath);
    float focalLength = 0.0f;
    if (argc > 2) {
        std::vector<float> R; // vector of rotation matrix of camera
        std::vector<float> O; // vector of translation matrix of camera
        std::string sfmFilePath = argv[2];
        std::string viewId = argv[3];
        readSFMData(sfmFilePath, viewId, R, O, focalLength);
        std::cout << "Focal length: " << focalLength << std::endl;
        // Reshape R to 3x3 matrix
        if (R.size() != 9) {
            std::cerr << "Error: R matrix should have 9 elements" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        cv::Mat R_mat = cv::Mat(3, 3, CV_32F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_mat.at<float>(i, j) = R[i * 3 + j];
            }
        }
        // Reshape O to 3X1 matrix
        if (O.size() != 3) {
            std::cerr << "Error: O matrix should have 3 elements" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        cv::Mat O_mat = cv::Mat(3, 1, CV_32F);
        for (int i = 0; i < 3; i++) {
            O_mat.at<float>(i, 0) = O[i];
        }
        cv::Mat T = cv::Mat::zeros(3, 3, CV_32F);
        T.at<float>(0, 0) = 1.0f;
        T.at<float>(1, 2) = 1.0f;   
        T.at<float>(2, 1) = -1.0f;
        cv::Mat temp = cv::Mat::zeros(3, 1, CV_32F);
        temp.at<float>(0, 2) = 1.0f;    
        cv::Mat directionMat = T * (R_mat * temp);

        temp.at<float>(0, 2) = 0.0f;
        temp.at<float>(0, 1) = -1.0f;
        cv::Mat upMat = T * (R_mat * temp);
        


        direction[0] = directionMat.at<float>(0, 0);
        direction[1] = directionMat.at<float>(1, 0);
        direction[2] = directionMat.at<float>(2, 0);
        up[0] = upMat.at<float>(0, 0);
        up[1] = upMat.at<float>(1, 0);
        up[2] = upMat.at<float>(2, 0);

        cv::Mat rightMat = directionMat.cross(upMat);
        right[0] = rightMat.at<float>(0, 0);
        right[1] = rightMat.at<float>(1, 0);
        right[2] = rightMat.at<float>(2, 0);

        std::cout << "Direction: " << direction[0] << " " << direction[1] << " " << direction[2] << std::endl;
        std::cout << "Up: " << up[0] << " " << up[1] << " " << up[2] << std::endl;  
        std::cout << "Right: " << right[0] << " " << right[1] << " " << right[2] << std::endl;  

        cv::Mat startPointMat = T * O_mat;
        startPoint[0] = startPointMat.at<float>(0, 0);
        startPoint[1] = startPointMat.at<float>(1, 0);
        startPoint[2] = startPointMat.at<float>(2, 0);

        std::cout << "Start point: " << startPoint[0] << " " << startPoint[1] << " " << startPoint[2] << std::endl;
        std::string imagePath = argv[4];  
        mask = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (mask.empty()) {
            std::cerr << "Error: could not open image file " << imagePath << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (mask.channels() != 1) {
            std::cerr << "Error: image should be grayscale" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    // process the mask
    std::vector<int> yindices;
    std::vector<int> xindices;
    if (!mask.empty()) {
        std::vector<cv::Point> nonZeroCoordinates;
        cv::findNonZero(mask, nonZeroCoordinates);
        for(const auto & point: nonZeroCoordinates) {
            xindices.push_back(point.x);
            yindices.push_back(point.y);
        }
    }

    
    // Read the number of lines in the file
    
    std::string line;
    float x, y, z;
    std::istringstream iss;
    int numV = 0;
    int currentV = 0;
    float3 * V = nullptr;
    int numF = 0;
    int currentF = 0;
    std::string tempFace1;
    std::string tempFace2;
    std::string tempFace3;
    uint3 * F = nullptr;
    float * h_areas = nullptr;



    while (std::getline(file, line)) {
        if(line[0] == '#') {
            if(numV == 0 && isVertexPositionsLine(line)) {
                std::string hash, secondToken;
                iss.clear();
                iss.str(line);
                iss >> hash >> secondToken;
                numV = std::stoi(secondToken);
                V = new float3[numV];    
                std::cout << "Number of vertices: " << numV << std::endl;
            } else if (numF == 0 && isMeshFacesLine(line)) {
                std::string hash, secondToken;
                iss.clear();
                iss.str(line);
                iss >> hash >> hash >> hash >> hash >> secondToken;
                numF = std::stoi(secondToken);
                F = new uint3[numF];
                h_areas = new float[numF]; 
                std::cout << "Number of faces: " << numF << std::endl;
            }
        } else {
            if (line.substr(0, 2) == "v "){
                iss.clear();
                iss.str(line.substr(2));
                iss >> x >> y >> z;
                V[currentV] = make_float3(x, y, z);
                currentV++;
            } else if (line.substr(0, 3) == "f  ") {
                iss.clear();
                iss.str(line.substr(3));
                iss >> tempFace1 >> tempFace2 >> tempFace3;
                F[currentF] = make_uint3(std::stoul(tempFace1.substr(0, tempFace1.find('/'))) - 1, std::stoul(tempFace2.substr(0, tempFace2.find('/'))) - 1, std::stoul(tempFace3.substr(0, tempFace3.find('/'))) - 1);
                currentF++;
            }
        }
    }
    file.close();
    float totalArea = 0.0f;

    float3 * d_V;
    uint3 * d_F;
    float * d_areas, * d_output;
    int* d_yindices;
    int* d_xindices;
    cudaMalloc((void**)&d_yindices, yindices.size() * sizeof(int));
    cudaMalloc((void**)&d_xindices, xindices.size() * sizeof(int));
    // Malloc memory for the device
    cudaMalloc((void**)&d_V, numV * sizeof(float3));
    cudaMalloc((void**)&d_F, numF * sizeof(uint3));
    cudaMalloc((void**)&d_areas, numF * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float)); 
    // Copy input data from host to device     
    cudaMemcpy(d_V, V, numV * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, numF * sizeof(uint3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yindices, yindices.data(), yindices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xindices, xindices.data(), xindices.size() * sizeof(int), cudaMemcpyHostToDevice);
    // Memory set to 0
    if (mask.empty()) {
        cudaMemset(d_output, 0.0f, sizeof(float));
        int blockSize = BLOCK_SIZE; // TODO: set the block size to 1024
        int numBlocks = (numF + blockSize - 1) / blockSize;    
        computeTriangleAreas<<<numBlocks, blockSize>>>(d_V, d_F, d_areas, numF);
        reduceSum<<<numBlocks, blockSize>>>(d_areas, d_output, numF);
        
        cudaMemcpy(&totalArea, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Total area: " << totalArea << std::endl;
    } else {
        cudaMemset(d_output, 0.0f, sizeof(float));
        int blockSize = BLOCK_SIZE; // TODO: set the block size to 1024
        int numBlocks = (numF + blockSize - 1) / blockSize;
        cudaMemcpyToSymbol(d_direction, direction.get(), 3 * sizeof(float));
        cudaMemcpyToSymbol(d_up, up.get(), 3 * sizeof(float));
        cudaMemcpyToSymbol(d_right, right.get(), 3 * sizeof(float));
        cudaMemcpyToSymbol(d_startPoint, startPoint.get(), 3 * sizeof(float));
        float sensorWidth = SENSOR_WIDTH;
        float camera[2] = {focalLength, sensorWidth}; // ! The order is important, always focal length first and then sensor width
        cudaMemcpyToSymbol(d_camera, camera, 2 * sizeof(float));
        float T[9] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f};
        cudaMemcpyToSymbol(d_rotation, T, 9 * sizeof(float));
        float NT[9] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
        cudaMemcpyToSymbol(d_nrotation, NT, 9 * sizeof(float));
        computeMaskArea3D<<<numBlocks, blockSize>>>(d_V, d_F, d_areas, numF, d_yindices, d_xindices, yindices.size());
        reduceSum<<<numBlocks, blockSize>>>(d_areas, d_output, numF);  
        cudaMemcpy(&totalArea, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        if(argc > 5){
            cudaMemcpy(h_areas, d_areas, numF * sizeof(float), cudaMemcpyDeviceToHost);
            std::string outputPath = argv[5];
            cropMesh(filePath, outputPath, h_areas);
        }
        std::cout << "Total area: " << totalArea << std::endl;
    }

    cudaFree(d_areas);
    cudaFree(d_V);
    cudaFree(d_F);
    cudaFree(d_output);
    cudaFree(d_yindices);
    cudaFree(d_xindices);
    delete[] V;
    delete[] F;
    delete[] h_areas;
    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " milliseconds" << std::endl;

    return 0;
    // Rest of your code here
}

#include <iostream>
#include <string>
#include "LiuHongCudaRayTracing.cuh"
#include <chrono>
#include "nlohmann/json.hpp"
#include <fstream>
#include <cassert>
#include <vector>

int main(int argc, char* argv[]) {
    // * read the json file for configuration **
    if(argc != 2){
        std::cout << "Usage: ./SkinMask configPath" << std::endl;
        return 1;
    }
    std::string configPath = argv[1];
    std::ifstream configFile(configPath);
    if(!configFile.is_open()){
        std::cerr << "Error: Cannot open file " << configPath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    nlohmann::json configData;
    configFile >> configData;
    configFile.close();

    // ************* Read SFM file *************
    auto start = std::chrono::high_resolution_clock::now();
    std::string inputObjPath = configData["inputObjPath"];
    LiuHongCudaRayTracing rayTracing = LiuHongCudaRayTracing();
    rayTracing.readObj(inputObjPath);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for reading obj file: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // ************* Build BVH ******************
    start = std::chrono::high_resolution_clock::now();
    rayTracing.buildBVH();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for building BVH: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // ************* Prepare Ray Tracing **************
    std::string sfmFilePath = configData["sfmFilePath"];
    std::vector<std::string> viewIds = configData["viewIds"].get<std::vector<std::string>>();
    std::string maskFolder = configData["maskFolder"];
    start = std::chrono::high_resolution_clock::now();
    rayTracing.prepareRaysGPU(maskFolder, viewIds, sfmFilePath);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for readMask: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    // ************* Ray Tracing ******************
    int threshold = configData["minCount"];
    std::string debugFolder = configData["debugFolder"];
    bool debugMode = configData["debug"];
    std::string groundTruthDepthPath = configData["groundTruthDepthPath"];
    std::string frameIdForDepth = configData["frameIdForDepth"];
    float minDepth = configData["minDepth"];
    float maxDepth = configData["maxDepth"];
    rayTracing.trimMeshGPU(maskFolder, 
        debugFolder, 
        inputObjPath, 
        threshold, 
        groundTruthDepthPath,
        frameIdForDepth,
        minDepth,
        maxDepth,
        debugMode); 

    return 0;
}
#pragma once 

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "nlohmann/json.hpp"
#include <memory>
#include <regex>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <cfloat>
struct Tri {float3 v0, v1, v2, centroids;};
struct BVHNode {
    float3 min, max;
    uint leftFirst, triCount;
};
class LiuHongCudaRayTracing {
    private:
        std::unique_ptr<float[]> direction;
        std::unique_ptr<float[]> startPoint;
        std::unique_ptr<float[]> up;
        std::unique_ptr<float[]> right;
        // std::unique_ptr<float[]> P0; 
        std::vector<std::string> validViewIds;
        float * h_areas = nullptr; // pointer to area for each face in host memory.
        int numF = 0;
        cv::Mat mask;
        // Obj GPU pre allocation
        float * d_areas, * d_areasSum;
        // BVH
        //struct Tri {float3 v0, v1, v2, centroids;};
        struct Ray {float3 O, D; float t = 1e30f;}; // ! Not use yet. 
        Tri * triangles = nullptr;
        uint * triIdx = nullptr;

        BVHNode * bvhNodes = nullptr;
        uint rootNodeIndex = 0, nodesUsed = 1;
        void _updateNodeBounds(uint nodeIdx);
        void _subdivideNode(uint nodeIdx);

        float focalLength;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        void _readSFMData(const std::string &sfmFilePath, const std::string & frameId, std::vector<float> &R, std::vector<float> &O);
        bool _isVertexPositionsLine(const std::string & line);
        bool _isMeshFacesLine(const std::string & line);
        void _cropMesh(const std::string & inputPath, const std::string & outputPath, const float * areas);
        uint _intersectBVH(const float3& O, const float3& D, const uint nodeIdx);
    public:
        LiuHongCudaRayTracing();
        void readObj(const std::string & objPath);
        void prepareRaysGPU(const std::string & maskFolder, const std::vector<std::string> & viewIds, const std::string & sfmFilePath);
        void trimMeshGPU(
            const std::string & maskFolder, 
            const std::string & debugFolder, 
            const std::string & objPath, 
            const int minNumMask,  
            const std::string & groundTruthDepthPath, const std::string & frameIdForDepth, 
            const float minDepth,
            const float maxDepth,
            const bool visualize = false);

        // ********************* BVH Public Functions ********************* //
        void buildBVH();
        void projectMask(const std::string & maskFolder);
        ~LiuHongCudaRayTracing();
};
#include "LiuHongCudaRayTracing.cuh"

#define BLOCK_SIZE 1024
#define SENSOR_WIDTH 36.0f
#define MaxNumViews 30
#define IMG_SIZE 1344
#define CUDA_CHECK_ERROR(call)                                                      \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;              \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }
// Define constant memory here
// ! Maximum number can be saved in constant memory is 64KB
__constant__ float d_direction[3 * MaxNumViews];
__constant__ float d_up[3 * MaxNumViews];
__constant__ float d_right[3 * MaxNumViews];
__constant__ float d_startPoint[3 * MaxNumViews];
// __constant__ float d_P0[3]; //Sensor points the camera focus on

// host device helper function
__host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// device level cross product function
__host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
// device level dot produc  t function
__host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
} 

// Möller–Trumbore intersection algorithm
__host__ __device__ float intersectTri(const float3& O, const float3& D, const Tri& tri){
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 P = cross(D, e2);
    float det = dot(e1, P);
    if (det > -1e-6 && det < 1e-6) {
        return -1.0f; // this ray is parallel to this triangle
    }
    float invDet = 1.0f / det;
    float3 T = O - tri.v0;
    float u = dot(T, P) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }
    float3 Q = cross(T, e1);
    float v = dot(D, Q) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }
    float t = dot(e2, Q) * invDet;
    if (t > 1e-6) {
        return t;
    }
    return -1.0f;
}

// Intersect AABB
__host__ __device__ bool intersectAABB(const float3& O, const float3 &D, const float3& bmin, const float3& bmax){
    float tx1 = (bmin.x - O.x) / D.x;
    float tx2 = (bmax.x - O.x) / D.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (bmin.y - O.y) / D.y;
    float ty2 = (bmax.y - O.y) / D.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - O.z) / D.z;
    float tz2 = (bmax.z - O.z) / D.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
    return tmax >= tmin && tmax > 0;
}

// compute the triangle Areas for overall faces
__global__ void computeTriangleAreas(Tri * triangles, float * areas, uint * hitCount, int numF, const int minNumMask){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numF){
        areas[idx] = 0.0f; //! very important, reset the areas to zeros.
    }
    if(idx < numF && hitCount[idx] >= minNumMask){
        float3 e1 = triangles[idx].v1 - triangles[idx].v0;
        float3 e2 = triangles[idx].v2 - triangles[idx].v0;
        float3 crossProduct = cross(e1, e2);
        float area = 0.5f * sqrtf(crossProduct.x * crossProduct.x + crossProduct.y * crossProduct.y + crossProduct.z * crossProduct.z);
        areas[idx] = area;
    }
}

// reduce the sum of all areas
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

// * study this method, it eat too many resources, register usage is too high 2024/10/23
__global__ void bvh3DBackTrack(const BVHNode * bvhNodes, const Tri * triangles, const uint * triIdx, const unsigned char * mask, float * depth, float minDepth, float maxDepth, float focalLength, int i){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < IMG_SIZE * IMG_SIZE){
        float3 O = make_float3(d_startPoint[3 * i], d_startPoint[3 * i + 1], d_startPoint[3 * i + 2]);
        float3 direction = make_float3(d_direction[3 * i], d_direction[3 * i + 1], d_direction[3 * i + 2]);
        float3 right = make_float3(d_right[3 * i], d_right[3 * i + 1], d_right[3 * i + 2]);
        float3 up = make_float3(d_up[3 * i], d_up[3 * i + 1], d_up[3 * i + 2]);
        if(true){
            if(depth[idx] > maxDepth || depth[idx] < minDepth){
                depth[idx] = 0.0f;
            } else {
                int xIndex = idx % IMG_SIZE;
                int yIndex = idx / IMG_SIZE;
                float x = (xIndex - IMG_SIZE / 2.f) * SENSOR_WIDTH / IMG_SIZE;
                float y = (IMG_SIZE / 2.f - yIndex) * SENSOR_WIDTH / IMG_SIZE;
                float3 D = make_float3(focalLength * direction.x + x * right.x + y * up.x, focalLength * direction.y + x * right.y + y * up.y, focalLength * direction.z + x * right.z + y * up.z);
                float D_length = sqrtf(D.x * D.x + D.y * D.y + D.z * D.z);
                // start bvh traversal
                uint stack[64];
                float distance = FLT_MAX;
                uint * stackPtr = stack;
                *stackPtr++ = 0;
                while(stackPtr > stack){
                    uint currentNodeIdx = *--stackPtr;
                    const BVHNode node = bvhNodes[currentNodeIdx];
                    if(!intersectAABB(O, D, node.min, node.max)){
                        continue;
                    }
                    if(node.triCount > 0){
                        for(int j = 0; j < node.triCount; j++){
                            uint temp = triIdx[node.leftFirst + j];
                            float t = intersectTri(O, D, triangles[temp]);
                            if(t > 0){
                                distance = fminf(distance, t * D_length);
                            }
                        }
                    } else {
                        *stackPtr++ = node.leftFirst;
                        *stackPtr++ = node.leftFirst + 1;
                    }
                }
                if(distance < FLT_MAX){
                    depth[idx] = depth[idx] / distance;
                } else {
                    depth[idx] = 0.0f;
                }
            }
        } else {
            depth[idx] = 0.0f;
        }
    }
}

// * bvh for mesh to mask ******
__global__ void bvh3DBackTrack(const BVHNode * bvhNodes, const Tri * triangles, const uint * triIdx, bool * blocked, int numF, int numFrames){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numF){
        float3 P1 = triangles[idx].centroids;
        uint stack[32]; //! 64 is LiuHong's estimation, if in the future we out of memory or we have larger mesh, change this number
        uint * stackPtr;
        for(int i = 0; i < numFrames; i++){
            bool isBlocked = false; // default we assume the ray is not blocked
            // * Traversal the BVH tree *
            stackPtr = stack;
            float3 P2 = make_float3(d_startPoint[3 * i], d_startPoint[3 * i + 1], d_startPoint[3 * i + 2]);
            float3 D = P2 - P1;
            float3 O = P1 - make_float3(1e-5f * D.x, 1e-5f * D.y, 1e-5f * D.z); // ! temporary fix by use very small number.
            *stackPtr++ = 0;
            while(stackPtr > stack && !isBlocked){
                uint currentNodeIdx = *--stackPtr;
                const BVHNode node = bvhNodes[currentNodeIdx];
                if(!intersectAABB(O, D, node.min, node.max)){
                    continue;
                }
                if(node.triCount > 0){ // means this is leaf node.
                    for(int j = 0; j < node.triCount && !isBlocked; j++){
                        uint temp = triIdx[node.leftFirst + j];
                        if(temp != idx && intersectTri(O, D, triangles[temp]) > 0){
                            // blocked
                            isBlocked = true;
                        }
                    }
                } else {
                    *stackPtr++ = node.leftFirst;
                    *stackPtr++ = node.leftFirst + 1;
                }
            }
            blocked[i * numF + idx] = isBlocked;
        }
    }
}

//Depth only global function
__global__ void maskRayTraceDepth(const unsigned char * mask, const bool * blocked, const float * depth, float * scales, const Tri * triangles, int numF, float focalLength, int idexDepthFrame, float minDepth, float maxDepth){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numF){
        scales[idx] = 0.0f; // initialize the scales
        float3 P1 = triangles[idx].centroids; // the center of mesh
        float3 P2 = make_float3(d_startPoint[3 * idexDepthFrame], d_startPoint[3 * idexDepthFrame + 1], d_startPoint[3 * idexDepthFrame + 2]); // the center of camera
        float3 D = P2 - P1; //vector from mesh point to camera center
        if(!blocked[idexDepthFrame * numF + idx]){ // if this triangle is not blocked by other triangles to this camera
            float3 n = make_float3(d_direction[3 * idexDepthFrame], d_direction[3 * idexDepthFrame + 1], d_direction[3 * idexDepthFrame + 2]); // direction of camera
            float3 P0 = P2 + make_float3(n.x * focalLength, n.y * focalLength, n.z * focalLength); // get the point center of the camera on sensor
            float denorm = dot(n, D); // get the length of ray on camera direction
            if(fabs(denorm) > 1e-6){ // make sure this ray is not parallel with camera direction
                float3 P1_0 = P0 - P1; //vector from P1 to P0
                float t = dot(n, P1_0) / denorm; // get the length of ray from P1 to intersection point
                float3 intersectionPoint = P1 + make_float3(t * D.x, t * D.y, t * D.z); // get the intersection point on camera sensor
                float3 P0_inter = intersectionPoint - P0; // This vecotr is on camera sensor
                float3 x_unit = make_float3(d_right[3 * idexDepthFrame], d_right[3 * idexDepthFrame + 1], d_right[3 * idexDepthFrame + 2]); // x unit vector of sensor
                float3 y_unit = make_float3(d_up[3 * idexDepthFrame], d_up[3 * idexDepthFrame + 1], d_up[3 * idexDepthFrame + 2]); // y unit vector of sensor
                float x = dot(P0_inter, x_unit); // project the intersection vector on sensor to x_unit vector, get the length of x coordinate
                float y = dot(P0_inter, y_unit); // project the intersection vector on sensor to y_unit vector, get the length of y coordinate
                int xIndex = (int)(x * float(IMG_SIZE) / SENSOR_WIDTH + IMG_SIZE / 2.f); // get the x index of the intersection point on sensor
                int yIndex = (int)(IMG_SIZE / 2.f - y * float(IMG_SIZE) / SENSOR_WIDTH); // get the y index of the intersection point on sensor
                float pixelDistance = sqrtf(focalLength * focalLength + x * x + y * y); // get the distance from mesh point to camera center
                if(xIndex >= 0 && xIndex < int(IMG_SIZE) && yIndex >= 0 && yIndex < int(IMG_SIZE)){ // make sure the intersection point is on the sensor
                    if(mask[yIndex * IMG_SIZE + xIndex] > 0){ // make sure the intersection point is on the mask
                        float depthValue = depth[yIndex * IMG_SIZE + xIndex]; // get the depth value of the intersection point
                        if(depthValue > minDepth && depthValue < maxDepth){ // make sure the depth value is in the range
                            scales[idx] = (depthValue - pixelDistance) / sqrtf(D.x * D.x + D.y * D.y + D.z * D.z) ;// set the scale to 1
                        }
                    }
                }
            }
        }
    }
}


__global__ void maskRayTrace(const unsigned char * masks, const bool * blocked, uint * hitCount, const Tri * triangles, int numF, int numFrames, float focalLength){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numF){
        hitCount[idx] = 0; //! very important to reset the hit count to 0 for each triangle
        float3 P1 = triangles[idx].centroids;
        for(int i = 0; i < numFrames; i++){
            float3 P2 = make_float3(d_startPoint[3 * i], d_startPoint[3 * i + 1], d_startPoint[3 * i + 2]);
            float3 D = P2 - P1;
            // * Check if this triangle is visible by camera *
            if(!blocked[i * numF + idx]){
                float3 n = make_float3(d_direction[3 * i], d_direction[3 * i + 1], d_direction[3 * i + 2]);
                float3 P0 = P2 + make_float3(n.x * focalLength, n.y * focalLength, n.z * focalLength);
                float denorm = dot(n, D);
                if(fabs(denorm) > 1e-6){
                    float3 P0_P1 = P0 - P1;
                    float t = dot(n, P0_P1) / denorm;
                    float3 intersectionPoint = P1 + make_float3(t * D.x, t * D.y, t * D.z);
                    float3 P02Intersection = intersectionPoint - P0;
                    float3 x_unit = make_float3(d_right[3 * i], d_right[3 * i + 1], d_right[3 * i + 2]);
                    float3 y_unit = make_float3(d_up[3 * i], d_up[3 * i + 1], d_up[3 * i + 2]);
                    float x = dot(P02Intersection, x_unit);
                    float y = dot(P02Intersection, y_unit);
                    int xIndex = (int)(x * float(IMG_SIZE) / SENSOR_WIDTH + IMG_SIZE / 2.f);
                    int yIndex = (int)(IMG_SIZE / 2.f - y * float(IMG_SIZE) / SENSOR_WIDTH);
                    if(xIndex >= 0 && xIndex < int(IMG_SIZE) && yIndex >= 0 && yIndex < int(IMG_SIZE)){
                        if(masks[i * IMG_SIZE * IMG_SIZE + yIndex * IMG_SIZE + xIndex] > 0){
                            hitCount[idx]++;
                        }
                    }
                }
            }
        }
    }
}

// Constructor
LiuHongCudaRayTracing::LiuHongCudaRayTracing() {
    direction = std::make_unique<float[]>(3 * MaxNumViews);
    std::fill(direction.get(), direction.get() + 3 * MaxNumViews, 0.0f);
    startPoint = std::make_unique<float[]>(3 * MaxNumViews);
    std::fill(startPoint.get(), startPoint.get() + 3 * MaxNumViews, 0.0f);
    up = std::make_unique<float[]>(3 * MaxNumViews);
    std::fill(up.get(), up.get() + 3 * MaxNumViews, 0.0f);
    right = std::make_unique<float[]>(3 * MaxNumViews);
    std::fill(right.get(), right.get() + 3 * MaxNumViews, 0.0f);
}

// Destructor
LiuHongCudaRayTracing::~LiuHongCudaRayTracing() {
    delete[] h_areas;
    delete[] triangles;
    delete[] triIdx;
    cudaFree(d_areas);
    cudaFree(d_areasSum);
}

void LiuHongCudaRayTracing::readObj(const std::string & objPath){
    std::string line;
    float x, y, z;
    std::istringstream iss;
    int currentV = 0;
    int currentF = 0;
    std::string tempFace1;
    std::string tempFace2;
    std::string tempFace3;
    int numV = 0;
    float3 * V = nullptr; // pointer to float3 vertex
    std::ifstream file(objPath);
    if(!file.is_open()){
        std::cerr << "Error: Cannot open file " << objPath << std::endl;
        std::exit(EXIT_FAILURE);
    } 
    while(std::getline(file, line)){
        if(line[0] == '#'){
            if(numV == 0 && _isVertexPositionsLine(line)){
                std::string hash, secondToken;
                iss.clear();
                iss.str(line);
                iss >> hash >> secondToken;
                numV = std::stoi(secondToken);
                V = new float3[numV];    
                std::cout << "Number of vertices: " << numV << std::endl;
            } else if (numF == 0 && _isMeshFacesLine(line)){
                std::string hash, secondToken;
                iss.clear();
                iss.str(line);
                iss >> hash >> hash >> hash >> hash >> secondToken;
                numF = std::stoi(secondToken);
                triangles = new Tri[numF];
                h_areas = new float[numF]; 
                std::cout << "Number of faces: " << numF << std::endl;
            }
        } else {
            if(line.substr(0,2) == "v "){
                iss.clear();
                iss.str(line.substr(2));
                iss >> x >> y >> z;
                V[currentV] = make_float3(x, -z, y);
                currentV++;
            } else if (line.substr(0, 3) == "f  ") {
                iss.clear();
                iss.str(line.substr(3));
                iss >> tempFace1 >> tempFace2 >> tempFace3;
                uint3 temp = make_uint3(std::stoul(tempFace1.substr(0, tempFace1.find('/'))) - 1, std::stoul(tempFace2.substr(0, tempFace2.find('/'))) - 1, std::stoul(tempFace3.substr(0, tempFace3.find('/'))) - 1);
                triangles[currentF].v0 = V[temp.x];
                triangles[currentF].v1 = V[temp.y];
                triangles[currentF].v2 = V[temp.z];
                triangles[currentF].centroids = make_float3((triangles[currentF].v0.x + triangles[currentF].v1.x + triangles[currentF].v2.x) / 3.0f, (triangles[currentF].v0.y + triangles[currentF].v1.y + triangles[currentF].v2.y) / 3.0f, (triangles[currentF].v0.z + triangles[currentF].v1.z + triangles[currentF].v2.z) / 3.0f);
                currentF++;
            }
        }
    }  
    file.close();
    delete[] V;
}

// private function to check if the line is vertex positions line
bool LiuHongCudaRayTracing::_isVertexPositionsLine(const std::string & line){
    std::regex pattern(R"(^# \d+ vertex positions$)");
    return std::regex_match(line, pattern);
}

void LiuHongCudaRayTracing::_cropMesh(const std::string & inputPath, const std::string & outputPath, const float * areas){
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

// private function to check if the line is mesh faces line
bool LiuHongCudaRayTracing::_isMeshFacesLine(const std::string & line){
    std::regex pattern(R"(^# Mesh '' with \d+ faces$)");
    return std::regex_match(line, pattern);
}


void LiuHongCudaRayTracing::prepareRaysGPU(const std::string & maskFolder, const std::vector<std::string> & viewIds, const std::string & sfmFilePath){
    // *************** Load SFM data *******************
    int numMasks = viewIds.size();
    std::cout << "Number of provided masks: " << numMasks << std::endl;
    std::ifstream file(sfmFilePath);
    if(!file.is_open()){
        std::cerr << "Error: Cannot open file " << sfmFilePath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    nlohmann::json jsonData;
    file >> jsonData;
    file.close();
    const auto & views = jsonData["views"];
    // for(const auto & view : views){
    //     std::string viewId_temp = view["viewId"];
    //     std::string frameId_temp = view["frameId"];
    //     frameId2viewId[frameId_temp] = viewId_temp;
    // }
    // * Load Camera intrinsic information *
    std::string fl = jsonData["intrinsics"][0]["focalLength"];
    focalLength = std::stof(fl);
    std::string w = jsonData["intrinsics"][0]["width"];
    float width = std::stof(w);
    std::string h = jsonData["intrinsics"][0]["height"];
    float height = std::stof(h);
    std::string xoffset = jsonData["intrinsics"][0]["principalPoint"][0];
    std::string yoffset = jsonData["intrinsics"][0]["principalPoint"][1];
    float cx = width / 2.0f + std::stof(xoffset);
    float cy = height / 2.0f + std::stof(yoffset);
    cameraMatrix = (cv::Mat_<float>(3, 3) << focalLength * width / SENSOR_WIDTH, 0.0f, cx, 0.0f, focalLength * height / SENSOR_WIDTH, cy, 0.0f, 0.0f, 1.0f);
    std::cout << "Camera matrix (SFM): " << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::string k1 = jsonData["intrinsics"][0]["distortionParams"][0];
    float k1f = std::stof(k1);
    std::string k2 = jsonData["intrinsics"][0]["distortionParams"][1];
    float k2f = std::stof(k2);
    std::string k3 = jsonData["intrinsics"][0]["distortionParams"][2];
    float k3f = std::stof(k3);
    distCoeffs = (cv::Mat_<float>(5, 1) << k1f, k2f, 0.0f, 0.0f, k3f);
    std::cout << "Distortion coefficients (SFM): " << std::endl;
    std::cout << distCoeffs << std::endl;
    // * Load camera Extrinsic information *
    const auto & poses = jsonData["poses"];
    cv::Mat T = cv::Mat::zeros(3, 3, CV_32F);
    T.at<float>(0,0) = 1.0f;
    T.at<float>(1,2) = 1.0f;
    T.at<float>(2,1) = -1.0f;
    cv::Mat dir_temp = cv::Mat::zeros(3, 1, CV_32F);
    dir_temp.at<float>(0,2) = 1.0f;
    cv::Mat up_temp = cv::Mat::zeros(3, 1, CV_32F);
    up_temp.at<float>(0,1) = -1.0f;
    int processedFrameCount = 0;
    for(const auto & viewId: viewIds){
        bool poseFound = false;
        for(const auto & pose: poses){
            if(pose["poseId"] == viewId){
                validViewIds.push_back(viewId);
                poseFound = true;
                std::vector<float> R;
                std::vector<float> O;
                const auto & rotation = pose["pose"]["transform"]["rotation"];
                for (const std::string r : rotation) {
                    R.push_back(std::stof(r));
                }
                const auto & translation = pose["pose"]["transform"]["center"];
                for (const std::string o : translation) {
                    O.push_back(std::stof(o));
                }
                if(R.size() != 9 || O.size() != 3){
                    std::cerr << "Error: Invalid SFM data" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                cv::Mat R_mat = cv::Mat(3, 3, CV_32F, R.data());
                cv::Mat O_mat = cv::Mat(3, 1, CV_32F, O.data());
                std::cout << "Rotation matrix for viewId " << viewId << ": " << std::endl;
                std::cout << R_mat << std::endl;
                std::cout << "Origin for viewId " << viewId << ": " << std::endl;
                std::cout << O_mat << std::endl;
                cv::Mat directionMat = T * (R_mat * dir_temp);
                cv::Mat upMat = T * (R_mat * up_temp);
                direction[processedFrameCount * 3] = directionMat.at<float>(0,0);
                direction[processedFrameCount * 3 + 1] = directionMat.at<float>(1,0);
                direction[processedFrameCount * 3 + 2] = directionMat.at<float>(2,0);
                up[processedFrameCount * 3] = upMat.at<float>(0,0);
                up[processedFrameCount * 3 + 1] = upMat.at<float>(1,0);
                up[processedFrameCount * 3 + 2] = upMat.at<float>(2,0);
                cv::Mat rightMat = directionMat.cross(upMat);
                right[processedFrameCount * 3] = rightMat.at<float>(0,0);
                right[processedFrameCount * 3 + 1] = rightMat.at<float>(1,0);
                right[processedFrameCount * 3 + 2] = rightMat.at<float>(2,0);
                cv::Mat startPointMat = T * O_mat;
                startPoint[processedFrameCount * 3] = startPointMat.at<float>(0,0);
                startPoint[processedFrameCount * 3 + 1] = startPointMat.at<float>(1,0);
                startPoint[processedFrameCount * 3 + 2] = startPointMat.at<float>(2,0);

                processedFrameCount++;

            }
        }
        if(!poseFound){
            std::cout << "Warning: Pose not found for frameId " << viewId << std::endl;
            std::cout << "Skipping frameId " << viewId << std::endl;
        }
    }
    std::cout << "Valid view Id: ";
    for(const auto & viewId: validViewIds){
        std::cout << viewId << " ";
    }
    std::cout << std::endl;
    std::cout << "Direction: ";
    for(int i = 0; i < processedFrameCount; i++){
        std::cout << "("<< direction[i * 3] << " " << direction[i * 3 + 1] << " " << direction[i * 3 + 2] << ") ";
    }
    std::cout << std::endl;
    std::cout << "Up: ";
    for(int i = 0; i < processedFrameCount; i++){
        std::cout << "("<< up[i * 3] << " " << up[i * 3 + 1] << " " << up[i * 3 + 2] << ") ";
    }
    std::cout << std::endl;
    std::cout << "Right: ";
    for(int i = 0; i < processedFrameCount; i++){
        std::cout << "("<< right[i * 3] << " " << right[i * 3 + 1] << " " << right[i * 3 + 2] << ") ";
    }
    std::cout << std::endl;
    std::cout << "Start point: ";
    for(int i = 0; i < processedFrameCount; i++){
        std::cout << "Start point: (" << startPoint[i * 3] << " " << startPoint[i * 3 + 1] << " " << startPoint[i * 3 + 2] << ")";
    }
    std::cout << std::endl;
}

void LiuHongCudaRayTracing::trimMeshGPU(
    const std::string & maskFolder, 
    const std::string & debugFolder, 
    const std::string & objPath, 
    const int minNumMask,  
    const std::string & groundTruthDepthPath, 
    const std::string & frameIdForDepth, 
    const float minDepth,
    const float maxDepth,
    const bool visualize){
    // * Load all mask images * 
    auto start = std::chrono::high_resolution_clock::now();
    int numFrames = validViewIds.size();
    unsigned char * h_masks;
    unsigned char * d_masks;
    float * h_groundTruthDepth = nullptr;
    float * d_groundTruthDepth;
    unsigned char * h_depthMask = nullptr;
    unsigned char * d_depthMask;
    float * h_scales = nullptr; // for save scale information
    float * d_scales;
    bool * d_blocked;
    BVHNode * d_bvhNodes;
    Tri * d_triangles;
    uint * d_triIdx;
    uint * d_hitCount;
    int * h_hitCount;
    int height;
    int step;
    int indexDepthFrame;
    //********* Read ground truth depth image */
    cv::Mat groundTruthDepth = cv::imread(groundTruthDepthPath, cv::IMREAD_UNCHANGED);
    if(groundTruthDepth.empty()){
        std::cerr << "Error: Cannot read ground truth depth image " << groundTruthDepthPath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if(groundTruthDepth.channels() != 1){
        std::cerr << "Error: image should be grayscale" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (groundTruthDepth.type() != CV_32FC1) {
        std::cerr << "Error: Ground truth depth image should be of type CV_32FC1 (float32)" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cv::Mat undistorted_groundTruthDepth;
    cv::undistort(groundTruthDepth, undistorted_groundTruthDepth, cameraMatrix, distCoeffs);
    h_groundTruthDepth = new float[IMG_SIZE * IMG_SIZE];
    std::memcpy(h_groundTruthDepth, undistorted_groundTruthDepth.ptr(), IMG_SIZE * IMG_SIZE * sizeof(float));

    // TODO: we can use openmp to accelerate this part.
    for(int i = 0; i < numFrames; i++){
        cv::Mat mask = cv::imread(maskFolder + "/" + validViewIds[i] + ".png", cv::IMREAD_UNCHANGED);
        if(mask.empty()){
            std::cerr << "Error: Cannot read mask image " << maskFolder + "/" + validViewIds[i] + ".png" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(mask.channels() != 1){
            std::cerr << "Error: image should be grayscale" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(i == 0){
            height = mask.rows; // height 1344
            step = mask.step; // step 1344
            h_masks = new unsigned char[numFrames * height * step];
        }
        if(validViewIds[i] == frameIdForDepth){ 
            indexDepthFrame = i;
            h_scales = new float[numF];
            h_depthMask = new unsigned char[IMG_SIZE * IMG_SIZE];
            std::memcpy(h_depthMask, mask.ptr(), IMG_SIZE * IMG_SIZE * sizeof(unsigned char));
            std::cout << "Found depth frame Id: " << frameIdForDepth << " in SFM data" << std::endl;
        }
        std::memcpy(h_masks + i * height * step, mask.ptr(), height * step);
    }
    if(!h_depthMask){
        std::cerr << "Error: depth frame Id: " << frameIdForDepth << " is not a valid frame in SFM data" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // * Constant memory copy *
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_direction, direction.get(), 3 * numFrames * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_up, up.get(), 3 * numFrames * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_right, right.get(), 3 * numFrames * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_startPoint, startPoint.get(), 3 * numFrames * sizeof(float)));
    
    // * Allocate memory on device *
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_masks, numFrames * height * step));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_depthMask, height * step));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_groundTruthDepth, IMG_SIZE * IMG_SIZE * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_scales, numF * sizeof(float))); // for save scale information
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_blocked, numFrames * numF * sizeof(bool)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_bvhNodes, 2 * numF * sizeof(BVHNode)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_triangles, numF * sizeof(Tri)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_triIdx, numF * sizeof(uint)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_hitCount, numF * sizeof(uint)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_areas, numF * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_areasSum, sizeof(float)));

    // * Copy data to device *
    CUDA_CHECK_ERROR(cudaMemcpy(d_masks, h_masks, numFrames * height * step, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_depthMask, h_depthMask, height * step, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_groundTruthDepth, h_groundTruthDepth, IMG_SIZE * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_bvhNodes, bvhNodes, 2 * numF * sizeof(BVHNode), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_triangles, triangles, numF * sizeof(Tri), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_triIdx, triIdx, numF * sizeof(uint), cudaMemcpyHostToDevice));
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Transfer data to gpu: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    int blockSize = BLOCK_SIZE;
    int numBlocks = (numF + blockSize - 1) / blockSize; // ! can only run wave schedule in RTX A2000, but will run fully concurrent in RTX 4090 with more SMs.
    float totalArea = 0.0f;
    bvh3DBackTrack<<<numBlocks, blockSize>>>(d_bvhNodes, d_triangles, d_triIdx, d_blocked, numF, numFrames);
    maskRayTrace<<<numBlocks, blockSize>>>(d_masks, d_blocked, d_hitCount, d_triangles, numF, numFrames, focalLength);
    computeTriangleAreas<<<numBlocks, blockSize>>>(d_triangles, d_areas, d_hitCount, numF, minNumMask);
    reduceSum<<<numBlocks, blockSize>>>(d_areas, d_areasSum, numF);
    //** This part for depth ray tracing */ 
    maskRayTraceDepth<<<numBlocks, blockSize>>>(d_depthMask, d_blocked, d_groundTruthDepth, d_scales, d_triangles, numF, focalLength, indexDepthFrame, minDepth, maxDepth);
    cudaDeviceSynchronize(); // ! Brair here
    // ** end of depth ray tracing */
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for ray tracing in GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
  
    start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaMemcpy(&totalArea, d_areasSum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_scales, d_scales, numF * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Total area: " << totalArea << std::endl;

    // * Compute the scale 
    float sumScales = 0.0f;
    int count = 0;
    for(int i = 0; i < numF; i++){
        if(h_scales[i] > 0.0f){
            sumScales += h_scales[i];
            count++;
        }
    }
    if(count > 0){
        float meanScale = sumScales / count;
        std::cout << "MeanScale: " << meanScale << "mm" << std::endl;
    } else {
        std::cout << "No triangles with scale > 0 found." << std::endl;
    }
    // * End of computing scale
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for post processing of raytracing in host: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    h_hitCount = new int[numF];
    if(visualize){
        CUDA_CHECK_ERROR(cudaMemcpy(h_hitCount, d_hitCount, numF * sizeof(int), cudaMemcpyDeviceToHost));
        std::string outputPath = debugFolder + "/debug.obj";
        std::ofstream outputFile(outputPath);
        std::ifstream inputFile(objPath);
        if(!outputFile.is_open()){
            std::cerr << "Error: Cannot open output file " << outputPath << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string line;
        int index = 0;
        while(std::getline(inputFile, line)){
            if(line.substr(0, 3) == "f  ") {
                if(h_hitCount[index] < minNumMask){
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
    
    // * free up memory 
    cudaFree(d_masks);
    cudaFree(d_bvhNodes);
    cudaFree(d_triangles);
    cudaFree(d_triIdx);
    cudaFree(d_hitCount);
    cudaFree(d_blocked);
    cudaFree(d_depthMask);
    cudaFree(d_groundTruthDepth);
    cudaFree(d_scales);
    delete[] h_scales;
    delete[] h_groundTruthDepth;
    delete[] h_masks;
    delete[] h_hitCount;
    delete[] h_depthMask;
}

void LiuHongCudaRayTracing::buildBVH(){
    bvhNodes = new BVHNode[numF * 2]; 
    triIdx = new uint[numF];
    for(uint i = 0; i < numF; i++){
        triIdx[i] = i;
    }
    // asign all triangles to root nodes
    BVHNode& root = bvhNodes[rootNodeIndex];
    root.leftFirst = 0, root.triCount = numF;
    LiuHongCudaRayTracing::_updateNodeBounds(rootNodeIndex);
    std::cout << "After updated node bounds" << std::endl;
    std::cout << "Root node min: " << root.min.x << " " << root.min.y << " " << root.min.z << std::endl;
    std::cout << "Root node max: " << root.max.x << " " << root.max.y << " " << root.max.z << std::endl;
    LiuHongCudaRayTracing::_subdivideNode(rootNodeIndex);
}

// void LiuHongCudaRayTracing::projectMask(const std::string & maskFolder){ // This is debug function for CPU side
//     for(auto & frameId: validFrameIds){
//         std::vector<uint> hitIdx;
//         cv::Mat mask = cv::imread(maskFolder + "/" + frameId + ".jpg", cv::IMREAD_UNCHANGED);
//         if(mask.empty()){
//             std::cerr << "Error: Cannot read mask image: " << frameId << std::endl;
//             std::exit(EXIT_FAILURE);
//         }
//         if(mask.channels() != 1){
//             std::cerr << "Error: Image should be grayscale: " << frameId << std::endl;
//             std::exit(EXIT_FAILURE);
//         }
//         std::vector<int> yindices;
//         std::vector<int> xindices;
//         std::vector<cv::Point> nonZeroCoordinates;
//         cv::findNonZero(mask, nonZeroCoordinates);
//         for(const auto & point: nonZeroCoordinates){
//             yindices.push_back(point.y);
//             xindices.push_back(point.x);
//         }
//         int numIndices = yindices.size();
//         std::cout << "Number of non-zero pixels: " << numIndices << std::endl;
//         float3 temp1 = make_float3(direction[0] * focalLength, direction[1] * focalLength, direction[2] * focalLength);
//         for(int i = 0; i < numIndices; i++){
//             float dx = xindices[i] - 1344.f / 2.f;
//             float dy = 1344.f / 2.f - yindices[i];
//             dx *= 36.f/1344.f;
//             dy *= 36.f/1344.f;
//             float3 temp2 = make_float3(right[0] * dx, right[1] * dx, right[2] * dx);
//             float3 temp3 = make_float3(up[0] * dy, up[1] * dy, up[2] * dy);
//             float3 D = make_float3(temp1.x + temp2.x + temp3.x, temp1.y + temp2.y + temp3.y, temp1.z + temp2.z + temp3.z);
//             float magnitudeD = sqrtf(D.x * D.x + D.y * D.y + D.z * D.z);
//             D.x /= magnitudeD;
//             D.y /= magnitudeD;
//             D.z /= magnitudeD;
//             float3 O = make_float3(startPoint[0], startPoint[1], startPoint[2]);
//             uint result = LiuHongCudaRayTracing::_intersectBVH(O, D, 0);
//             hitIdx.push_back(result);
//         }
//         std::unordered_set<uint> uniqueHitIdx(hitIdx.begin(), hitIdx.end());
//         std::string inputPath = "/workdir/MeshroomCache/Texturing/66798f0b99a3b8d240d740f5f4456b533f8da933/texturedMesh.obj";
//         std::string outputPath = "/workdir/figure/cropped.obj";
//         std::ofstream outputFile(outputPath);
//         std::ifstream inputFile(inputPath);
//         if(!outputFile.is_open()){
//             std::cerr << "Error: could not open output file " << outputPath << std::endl;
//             std::exit(EXIT_FAILURE);
//         }
//         std::string line;
//         int index = 0;
//         while(std::getline(inputFile, line)){
//             if(line.substr(0, 3) == "f  ") {
//                 if(uniqueHitIdx.find(index) == uniqueHitIdx.end()) {
//                     index++;
//                     continue;
//                 } else {
//                     index++;
//                 }
//             }
//             outputFile << line << std::endl;
//         }
//         outputFile.close();
//     }

// }

uint LiuHongCudaRayTracing::_intersectBVH(const float3& O, const float3& D, const uint nodeIdx){
    BVHNode& node = bvhNodes[nodeIdx];
    if(!intersectAABB(O, D, node.min, node.max)) return 0;
    if(node.triCount > 0){
        for(uint i = 0; i < node.triCount; i++){
            uint idx = triIdx[node.leftFirst + i];
            if(intersectTri(O, D, triangles[idx])) return idx;
        }
        return 0;
    }else{
        uint result = 0;
        result = LiuHongCudaRayTracing::_intersectBVH(O, D, node.leftFirst);
        if(result > 0){
            return result;
        }
        result = LiuHongCudaRayTracing::_intersectBVH(O, D, node.leftFirst + 1);
        return result;
    }
}

void LiuHongCudaRayTracing::_updateNodeBounds(uint nodeIdx){
    BVHNode& node = bvhNodes[nodeIdx];
    node.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    node.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (uint first = node.leftFirst, i = 0; i < node.triCount; ++i) {
        uint leafTriIdx = triIdx[first + i];
        Tri& leafTri = triangles[leafTriIdx];
        node.min = make_float3(fminf(node.min.x, fminf(leafTri.v0.x, fminf(leafTri.v1.x, leafTri.v2.x))),
                               fminf(node.min.y, fminf(leafTri.v0.y, fminf(leafTri.v1.y, leafTri.v2.y))),
                               fminf(node.min.z, fminf(leafTri.v0.z, fminf(leafTri.v1.z, leafTri.v2.z))));
        node.max = make_float3(fmaxf(node.max.x, fmaxf(leafTri.v0.x, fmaxf(leafTri.v1.x, leafTri.v2.x))),
                                 fmaxf(node.max.y, fmaxf(leafTri.v0.y, fmaxf(leafTri.v1.y, leafTri.v2.y))),
                                 fmaxf(node.max.z, fmaxf(leafTri.v0.z, fmaxf(leafTri.v1.z, leafTri.v2.z))));
    }
}

void LiuHongCudaRayTracing::_subdivideNode(uint nodeIdx){
    BVHNode& node = bvhNodes[nodeIdx];
    if(node.triCount <= 2){
        return;
    }
    float3 extent = node.max - node.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;
    float splitPos;
    if(axis == 0){
        splitPos = node.min.x + extent.x * 0.5f;
    } else if (axis == 1)
    {
        splitPos = node.min.y + extent.y * 0.5f;
    } else {
        splitPos = node.min.z + extent.z * 0.5f;
    }
    int i = node.leftFirst, j = node.leftFirst + node.triCount - 1;
    while(i <= j){
        const float3& centroids = triangles[triIdx[i]].centroids;
        if(axis == 0){
            if (centroids.x < splitPos) {
                i++;
            } else {
                std::swap(triIdx[i], triIdx[j--]);
            }
        } else if (axis == 1){
            if (centroids.y < splitPos) {
                i++;
            } else {
                std::swap(triIdx[i], triIdx[j--]);
            }
        } else {
            if (centroids.z < splitPos) {
                i++;
            } else {
                std::swap(triIdx[i], triIdx[j--]);
            }
        } 
    }
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) {
        return;
    }
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    bvhNodes[leftChildIdx].leftFirst = node.leftFirst;
    bvhNodes[leftChildIdx].triCount = leftCount;
    bvhNodes[rightChildIdx].leftFirst = i;
    bvhNodes[rightChildIdx].triCount = node.triCount - leftCount;
    node.leftFirst = leftChildIdx;
	node.triCount = 0;
	LiuHongCudaRayTracing::_updateNodeBounds( leftChildIdx );
	LiuHongCudaRayTracing::_updateNodeBounds( rightChildIdx );
    LiuHongCudaRayTracing::_subdivideNode( leftChildIdx );
	LiuHongCudaRayTracing::_subdivideNode( rightChildIdx );
}
# Purpose of this repository
- This Repo I want to demonstrate how to use photogrammetry and cuda projection projects.

1. Utilize photogrammetry to create 3D obj mesh with color (Meshroom)

2. Once you have the 3D obj mesh, we can project any image mask from 2D pixel (scanned image user used for photogrammetry) to 3D mesh (obj file) by using BVH tree data structure for fast raycasting. 

3. Mesh voting mechanism is constructed inside GPU with CUDA streaming to further smooth the projection performance.

4. Unit convertion is conducted based on crestereo (you can also use Lidar), which calibrate each pixel to 3D mesh distance to real world units.

# 0. Environment setup
Use Dockerfile and NVIDIA Container Toolkit to build the docker environment. 
# 1. Photogrammetry
Meshroom take care of the photogrammetry process. 
- Input: 
    - N image scans
    - N masks (optional)
- Output: 
    - cameras.sfm (camera intrinsic and extrinsic parameters)
    - mesh.obj (3D mesh)
    - texture.jpg (color texture)
    - color wrapping information (uv map)
# 2. Generate 2D mask on each image.
You can use whatever method to generate the mask based your need on each image scan.
# 3. CUDA Projection
## 3.1 BVH tree construction. 
- Build BVH tree in CPU
    - Initialize BVH tree as BVHNode array
    - Assign triangle index to each triangle
    - Construct BVH tree recursively
```cpp
buildBVH(){
    bvhNodes = new BVHNode[numF * 2]; 
    triIdx = new uint[numF];
    for(uint i = 0; i < numF; i++){
        triIdx[i] = i;
    }
    BVHNode& root = bvhNodes[rootNodeIndex];
    root.leftFirst = 0, root.triCount = numF;
    updateNodeBounds(rootNodeIndex);
    subdivideNode(rootNodeIndex);
}
```
- Helper function for BVH tree construction
```cpp

```


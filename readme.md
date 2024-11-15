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
    - Update the Node Bounds AABB
    - Subdivide the Node to left and right child
```cpp
updateNodeBounds(uint nodeIdx){
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
```
```cpp
subdivideNode(uint nodeIdx){
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
	updateNodeBounds( leftChildIdx );

	updateNodeBounds( rightChildIdx );
    
    subdivideNode( leftChildIdx );

	subdivideNode( rightChildIdx );
}
```


